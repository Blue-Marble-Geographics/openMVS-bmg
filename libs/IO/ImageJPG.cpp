////////////////////////////////////////////////////////////////////
// ImageJPG.cpp
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#include "Common.h"

#ifdef _IMAGE_JPG
#include "ImageJPG.h"

#ifdef _MSC_VER
#define XMD_H // prevent redefinition of INT32
#undef FAR  // prevent FAR redefinition
#endif

extern "C" {
#include <jpeglib.h>
}
#include <setjmp.h>

using namespace SEACAVE;


// D E F I N E S ///////////////////////////////////////////////////

#define JPG_BUFFER_SIZE	(16*1024)

struct JpegErrorMgr
{
	struct jpeg_error_mgr pub;
	jmp_buf setjmp_buffer;
};

struct JpegSource
{
	struct jpeg_source_mgr pub;
	ISTREAM* pStream;
	JOCTET* buffer;
};

struct JpegState
{
	jpeg_decompress_struct cinfo; // IJG JPEG codec structure
	JpegErrorMgr jerr;// error processing manager state
	JpegSource source;// memory buffer source
};


// F U N C T I O N S ///////////////////////////////////////////////

METHODDEF(void)
stub(j_decompress_ptr cinfo)
{
	JpegSource* source = (JpegSource*)cinfo->src;
	source->pStream->setPos(0);
}

METHODDEF(boolean)
fill_input_buffer(j_decompress_ptr cinfo)
{
	JpegSource* source = (JpegSource*)cinfo->src;
	const size_t size = source->pStream->read(source->buffer, JPG_BUFFER_SIZE);
	if (size == STREAM_ERROR || size == 0)
	return FALSE;
	source->pub.next_input_byte = source->buffer;
	source->pub.bytes_in_buffer = size;
	return TRUE;
}

METHODDEF(void)
skip_input_data(j_decompress_ptr cinfo, long num_bytes)
{
	JpegSource* source = (JpegSource*)cinfo->src;

	if (num_bytes > (long)source->pub.bytes_in_buffer)
	{
		// We need to skip more data than we have in the buffer.
		source->pStream->setPos(source->pStream->getPos() + (num_bytes - source->pub.bytes_in_buffer));
		source->pub.next_input_byte += source->pub.bytes_in_buffer;
		source->pub.bytes_in_buffer = 0;
	}
	else
	{
		// Skip portion of the buffer
		source->pub.bytes_in_buffer -= num_bytes;
		source->pub.next_input_byte += num_bytes;
	}
}

METHODDEF(void)
error_exit(j_common_ptr cinfo)
{
	JpegErrorMgr* err_mgr = (JpegErrorMgr*)(cinfo->err);

	/* Return control to the setjmp point */
	longjmp( err_mgr->setjmp_buffer, 1 );
}


// S T R U C T S ///////////////////////////////////////////////////

CImageJPG::CImageJPG() : m_state(NULL)
{
} // Constructor

CImageJPG::~CImageJPG()
{
	//clean up
	Close();
} // Destructor
/*----------------------------------------------------------------*/

void CImageJPG::Close()
{
	if (m_state)
	{
		JpegState* state = (JpegState*)m_state;
		jpeg_destroy_decompress( &state->cinfo );
		delete state;
		m_state = NULL;
	}
	m_width = m_height = 0;
	CImage::Close();
}
/*----------------------------------------------------------------*/

HRESULT CImageJPG::ReadHeader()
{
	JpegState* state = new JpegState;
	m_state = state;
	state->cinfo.err = jpeg_std_error(&state->jerr.pub);
	state->jerr.pub.error_exit = error_exit;

	if (setjmp(state->jerr.setjmp_buffer ) == 0)
	{
		jpeg_create_decompress( &state->cinfo );

		// Prepare for suspending reader
		state->source.pub.init_source = stub;
		state->source.pub.fill_input_buffer = fill_input_buffer;
		state->source.pub.skip_input_data = skip_input_data;
		state->source.pub.resync_to_restart = jpeg_resync_to_restart;
		state->source.pub.term_source = stub;
		state->source.pub.bytes_in_buffer = 0;// forces fill_input_buffer on first read
		state->source.pub.next_input_byte = NULL;
		state->source.pStream = (IOSTREAM*)m_pStream;
		state->source.buffer = (JOCTET*)(*state->cinfo.mem->alloc_small)((j_common_ptr)&state->cinfo, JPOOL_PERMANENT, JPG_BUFFER_SIZE * sizeof(JOCTET));
		state->cinfo.src = &state->source.pub;

		jpeg_read_header(&state->cinfo, TRUE);

		m_dataWidth = m_width = state->cinfo.image_width;
		m_dataHeight= m_height = state->cinfo.image_height;
		m_numLevels = 0;
		m_level     = 0;
		m_stride    = state->cinfo.num_components;
		m_lineWidth = m_width * m_stride;
		switch (m_stride)
		{
		case 1:
			m_format = PF_GRAY8;
			state->cinfo.out_color_space = JCS_GRAYSCALE;
			state->cinfo.out_color_components = 1;
			break;
		case 3:
			m_format = PF_B8G8R8;
			state->cinfo.out_color_space = JCS_RGB;
			state->cinfo.out_color_components = 3;
			break;
		default:
			LOG(LT_IMAGE, "error: unsupported JPG image");
			return _INVALIDFILE;
		}
		return _OK;
	}

	Close();
	return _FAIL;
} // ReadHeader
/*----------------------------------------------------------------*/

// Scale in the DCT domain instead of decoding full-size and resampling afterwards.
// libjpeg emits ceil(dim * scale_num / scale_denom); at 1/2 the IDCT drops from 8x8
// to 4x4 and upsampling plus colour conversion run on a quarter of the pixels, so
// this is worth roughly 2x -- not 4x, because Huffman decoding is unchanged.
//
// Restricted to 1/8, 1/4, 1/2: denom 8 with a power-of-two num is supported by plain
// libjpeg 6b as well as libjpeg-turbo. turbo also accepts any num in 1..16, but the
// extra granularity is not worth depending on which flavour is linked.
//
// Picks the SMALLEST output that still covers nMaxResolution, so the caller's
// subsequent ResizeImage only ever downsamples -- never upsamples, which would throw
// away real resolution.
//
// NOT equivalent to a full decode followed by INTER_AREA: a DCT-domain reduction is a
// different low-pass than a box filter over full-resolution pixels. Pixel values
// differ, so the caller gates this off by default.
void CImageJPG::SetDecodeScale(Size nMaxResolution)
{
	JpegState* state = (JpegState*)m_state;
	if (state == NULL || nMaxResolution == 0 || m_width == 0 || m_height == 0)
		return;
	jpeg_decompress_struct* cinfo = &state->cinfo;
	const Size full = (m_width > m_height ? m_width : m_height);
	if (full <= nMaxResolution)
		return; // already at or below the target; nothing to gain
	// EXACT DIVISIBILITY IS REQUIRED, not just convenient. libjpeg emits
	// ceil(dim * num / 8), and the caller then resizes by nMaxResolution/decodedWidth.
	// If ceil() rounded either dimension up, that second ratio differs from the one
	// the full-resolution path would have used and the FINAL size can land one pixel
	// off -- which changes every plane dimension, the tile grid, and the decode-cache
	// key (`MAXF(cache.width, cache.height) != imageSize`) downstream. Requiring both
	// dimensions to divide exactly makes ceil() a no-op, so the final size is
	// bit-for-bit the size the old path produced. Camera JPEGs are almost always even,
	// so this rejects very little in practice and costs nothing when it does.
	const unsigned kNums[3] = { 1u, 2u, 4u }; // over 8 -> 1/8, 1/4, 1/2
	unsigned num = 8u;
	for (int i = 0; i < 3; ++i) {
		const unsigned den = 8u / kNums[i]; // 8, 4, 2
		if ((m_width % den) != 0 || (m_height % den) != 0)
			continue; // ceil() would round; skip this ratio
		const Size outMax = (full * kNums[i]) / 8;
		if (outMax >= nMaxResolution) { num = kNums[i]; break; }
	}
	if (num >= 8u)
		return; // no exact ratio both covers the target and divides evenly
	cinfo->scale_num = num;
	cinfo->scale_denom = 8;
	jpeg_calc_output_dimensions(cinfo);
	// ReadData loops m_height times and the caller allocated GetWidth() x GetHeight(),
	// so BOTH must become the post-scale size here or the decode walks off the buffer.
	m_dataWidth = m_width = cinfo->output_width;
	m_dataHeight = m_height = cinfo->output_height;
	m_lineWidth = m_width * m_stride;
} // SetDecodeScale
/*----------------------------------------------------------------*/

HRESULT CImageJPG::ReadData(void* pData, PIXELFORMAT dataFormat, Size nStride, Size lineWidth)
{
	JpegState* state = (JpegState*)m_state;

	if (state && m_width && m_height)
	{
		jpeg_decompress_struct* cinfo = &state->cinfo;
		JpegErrorMgr* jerr = &state->jerr;

		if (setjmp(jerr->setjmp_buffer) == 0)
		{
			jpeg_start_decompress(cinfo);

			// read data
			if (dataFormat == m_format && nStride == m_stride) {
				// read image directly to the data buffer
				JSAMPLE* buffer[1] = {(JSAMPLE*)pData};
				uint8_t*& data = (uint8_t*&)buffer[0];
				for (Size j=0; j<m_height; ++j, data+=lineWidth)
					jpeg_read_scanlines(cinfo, buffer, 1);
			} else {
				// read image to a buffer and convert it
				JSAMPARRAY buffer = (*cinfo->mem->alloc_sarray)((j_common_ptr)cinfo, JPOOL_IMAGE, m_lineWidth, 1);
				uint8_t* dst = (uint8_t*)pData;
				uint8_t* src = (uint8_t*)buffer[0];
				for (Size j=0; j<m_height; ++j, dst+=lineWidth) {
					jpeg_read_scanlines(cinfo, buffer, 1);
					if (!FilterFormat(dst, dataFormat, nStride, src, m_format, m_stride, m_width))
						return _FAIL;
				}
			}

			jpeg_finish_decompress(cinfo);
			return _OK;
		}
	}

	Close();
	return _FAIL;
} // Read
/*----------------------------------------------------------------*/

HRESULT CImageJPG::WriteHeader(PIXELFORMAT imageFormat, Size width, Size height, BYTE numLevels)
{
	//TODO: to implement the JPG encoder
	return _OK;
} // WriteHeader
/*----------------------------------------------------------------*/

HRESULT CImageJPG::WriteData(void* pData, PIXELFORMAT dataFormat, Size nStride, Size lineWidth)
{
	//TODO: to implement the JPG encoder
	//const int quality = 100;
	//struct jpeg_compress_struct cinfo;
	//struct jpeg_error_mgr jerr;
	///* More stuff */
	//FILE * outfile;		/* target file */
	//JSAMPROW row_pointer[1];	/* pointer to JSAMPLE row[s] */
	//int row_stride;		/* physical row width in image buffer */

	//cinfo.err = jpeg_std_error(&jerr);
	//jpeg_create_compress(&cinfo);

	//if ((outfile = fopen(filename.c_str(), "wb")) == NULL) {
	//	fprintf(stderr, "can't open %s\n", filename.c_str());
	//	exit(1);
	//}
	//jpeg_stdio_dest(&cinfo, outfile);

	//cinfo.image_width = width;
	//cinfo.image_height = height;
	//cinfo.input_components = 3;
	//cinfo.in_color_space = JCS_RGB;
	//jpeg_set_defaults(&cinfo);
	//jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);

	//jpeg_start_compress(&cinfo, TRUE);

	//row_stride = width * 3;	/* JSAMPLEs per row in image_buffer */

	//while (cinfo.next_scanline < cinfo.image_height) {
	//	if (flip)
	//		row_pointer[0] = (JSAMPROW)& buffer[(cinfo.image_height - 1 - cinfo.next_scanline) * row_stride];
	//	else
	//		row_pointer[0] = (JSAMPROW)& buffer[cinfo.next_scanline * row_stride];
	//	(void) jpeg_write_scanlines(&cinfo, row_pointer, 1);

	//}

	//jpeg_finish_compress(&cinfo);
	//fclose(outfile);

	//jpeg_destroy_compress(&cinfo);
	return _OK;
} // WriteData
/*----------------------------------------------------------------*/

#endif // _IMAGE_JPG
