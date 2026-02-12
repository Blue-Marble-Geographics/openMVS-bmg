////////////////////////////////////////////////////////////////////
// OBJ.cpp
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#include "Common.h"
#include "OBJ.h"

using namespace SEACAVE;


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define OBJ_USE_OPENMP
#endif

#define OBJ_INDEX_OFFSET 1
#define FASTER_OBJ

// S T R U C T S ///////////////////////////////////////////////////

ObjModel::MaterialLib::Material::Material(const Image8U3& _diffuse_map, const Color& _Kd)
	:
	diffuse_map(_diffuse_map),
	Kd(_Kd)
{
}

bool ObjModel::MaterialLib::Material::LoadDiffuseMap()
{
	if (diffuse_map.empty())
		return diffuse_map.Load(diffuse_name);
	return true;
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

ObjModel::MaterialLib::MaterialLib()
{
}

bool ObjModel::MaterialLib::SaveSingleMaterial(const String& prefix, bool texLossless) const
{
	const Material& mat = materials[0];

	std::ofstream out(prefix + ".mtl");
	if (!out.good())
		return false;

	const String pathName(Util::getFilePath(prefix));
	const String name(Util::getFileNameExt(prefix));

	String diffuseName = mat.diffuse_name;
	if (diffuseName.IsEmpty())
		diffuseName = name + "_" + mat.name + "_map_Kd." + (texLossless ? "png" : "jpg");

	// Write MTL once
	out
		<< "newmtl " << mat.name << "\n"
		<< "Ka 1.000000 1.000000 1.000000\n"
		<< "Kd " << mat.Kd.r << " " << mat.Kd.g << " " << mat.Kd.b << "\n"
		<< "Ks 0.000000 0.000000 0.000000\n"
		<< "Tr 1.000000\n"
		<< "illum 1\n"
		<< "Ns 1.000000\n";

	if (!mat.diffuse_map.empty()) {
		out << "map_Kd " << diffuseName << "\n";
		if (!mat.diffuse_map.Save(pathName + diffuseName))
			return false;
	}

	return true;
}

bool ObjModel::MaterialLib::SaveMultipleMaterials(const String& prefix, bool texLossless) const
{
	std::ofstream out((prefix+".mtl").c_str());
	if (!out.good())
		return false;

	const String pathName(Util::getFilePath(prefix));
	const String name(Util::getFileNameExt(prefix));
	#ifdef OBJ_USE_OPENMP
	bool bSuccess(true);
	#pragma omp parallel for
	#endif
	for (int_t i = 0; i < (int_t)materials.size(); ++i) {
		const Material& mat = materials[i];
		// save material description
		std::stringstream ss;
		ss << "newmtl " << mat.name << "\n"
			<< "Ka 1.000000 1.000000 1.000000" << "\n"
			<< "Kd " << mat.Kd.r << " " << mat.Kd.g << " " << mat.Kd.b << "\n"
			<< "Ks 0.000000 0.000000 0.000000" << "\n"
			<< "Tr 1.000000" << "\n"
			<< "illum 1" << "\n"
			<< "Ns 1.000000" << "\n";
		// save material maps
		if (mat.diffuse_map.empty()) {
			#ifdef OBJ_USE_OPENMP
			#pragma omp critical
			#endif
			out << ss.str();
			continue;
		}
		if (mat.diffuse_name.IsEmpty())
			const_cast<String&>(mat.diffuse_name) = name+"_"+mat.name+"_map_Kd."+(texLossless?"png":"jpg");
		ss << "map_Kd " << mat.diffuse_name << "\n";
		#ifdef OBJ_USE_OPENMP
		#pragma omp critical
		#endif
		out << ss.str();
		const bool bRet(mat.diffuse_map.Save(pathName+mat.diffuse_name));
		#ifdef OBJ_USE_OPENMP
		#pragma omp critical
		if (!bRet)
			bSuccess = false;
		#else
		if (!bRet)
			return false;
		#endif
	}
	#ifdef OBJ_USE_OPENMP
	return bSuccess;
	#else
	return true;
	#endif
}

bool ObjModel::MaterialLib::Save(const String& prefix, bool texLossless) const
{
	if (materials.size() == 1) {
		return SaveSingleMaterial(prefix, texLossless);
	}
	return SaveMultipleMaterials(prefix, texLossless);
}

bool ObjModel::MaterialLib::Load(const String& fileName)
{
	const size_t numMaterials(materials.size());
	std::ifstream in(fileName.c_str());
	String keyword;
	while (in.good() && in >> keyword) {
		if (keyword == "newmtl") {
			in >> keyword;
			materials.push_back(Material(keyword));
		} else if (keyword == "Kd") {
			ASSERT(numMaterials < materials.size());
			Color c;
			in >> c.r >> c.g >> c.b;
			materials.back().Kd = c;
		} else if (keyword == "map_Kd") {
			ASSERT(numMaterials < materials.size());
			String& diffuse_name = materials.back().diffuse_name;
			in >> diffuse_name;
			diffuse_name = Util::getFilePath(fileName) + diffuse_name;
		}
	}
	return numMaterials < materials.size();
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

#ifdef FASTER_OBJ

class RawFileWriter {
public:
	explicit RawFileWriter(const char* path) {
#ifdef _WIN32
		handle = CreateFileA(
			path,
			GENERIC_WRITE,
			0,
			nullptr,
			CREATE_ALWAYS,
			FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
			nullptr
		);
#else
		fd = ::open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
#endif
	}

	~RawFileWriter() {
#ifdef _WIN32
		if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
		if (fd >= 0) ::close(fd);
#endif
	}

	bool IsValid() const {
#ifdef _WIN32
		return handle != INVALID_HANDLE_VALUE;
#else
		return fd >= 0;
#endif
	}

	void Write(const char* data, size_t size) {
#ifdef _WIN32
		DWORD written;
		WriteFile(handle, data, (DWORD)size, &written, nullptr);
#else
		::write(fd, data, size);
#endif
	}

private:
#ifdef _WIN32
	HANDLE handle = INVALID_HANDLE_VALUE;
#else
	int fd = -1;
#endif
};

constexpr double MaxSafeInt = 18446744073709551615.0; // UINT64_MAX
// Static lookup table for pairs of digits
static const char DigitTable[] =
"00010203040506070809"
"10111213141516171819"
"20212223242526272829"
"30313233343536373839"
"40414243444546474849"
"50515253545556575859"
"60616263646566676869"
"70717273747576777879"
"80818283848586878889"
"90919293949596979899";

__forceinline char* WriteDouble(char* p, double v) noexcept {
	if (!std::isfinite(v)) {
		*p++ = '0'; return p;
	}

	if (v < 0.0) {
		*p++ = '-';
		v = -v;
	}

	// 1. Separate integer and fraction with rounding
	// Use 1e6 for 6 decimal places
	double integral;
	double fractional = std::modf(v + 0.0000005, &integral);
	uint64_t ip = (uint64_t)integral;
	uint32_t f = (uint32_t)(fractional * 1000000.0);

	// 2. Integer part (backward write)
	char temp[24];
	char* t = temp + 24;

	if (ip == 0) {
		*--t = '0';
	}
	else {
		while (ip >= 100) {
			const uint64_t idx = (ip % 100) << 1;
			ip /= 100;
			*--t = DigitTable[idx + 1];
			*--t = DigitTable[idx];
		}
		if (ip >= 10) {
			const uint64_t idx = ip << 1;
			*--t = DigitTable[idx + 1];
			*--t = DigitTable[idx];
		}
		else if (ip > 0) {
			*--t = (char)('0' + ip);
		}
	}

	size_t ipLen = (size_t)((temp + 24) - t);
	memcpy(p, t, ipLen);
	p += ipLen;

	// 3. Fractional part (only if non-zero)
	if (f > 0) {
		*p++ = '.';
		// Write 6 digits but you could optimize this to trim zeros
		uint32_t div = 100000;
		for (int i = 0; i < 6; ++i) {
			uint32_t d = f / div;
			*p++ = (char)('0' + d);
			f %= div;
			div /= 10;
		}
		// Optional: while (*(p-1) == '0') p--; // Trim trailing zeros
	}

	return p;
}

#include <charconv>

__forceinline char* WriteUInt(char* out, uint32_t v) {
	auto r = std::to_chars(out, out + 32, v);
	return r.ptr;
}

bool ObjModel::Save(const String& fileName, unsigned precision, bool texLossless) const
{
	if (vertices.empty())
		return false;

	const String prefix(Util::getFileFullName(fileName));
	const String name(Util::getFileNameExt(prefix));

	if (!material_lib.Save(prefix, texLossless))
		return false;

	const String objPath = prefix + ".obj";
	RawFileWriter file(objPath.c_str());
	if (!file.IsValid())
		return false;

	const int maxThreads = omp_get_max_threads();

	constexpr size_t kComponentSize = 32;
	constexpr size_t kVertexSize = 3 * kComponentSize + 5; // "v " + 3 components + "\n"
	constexpr size_t kTexCoordSize = 2 * kComponentSize + 5; // "vt " + 2 components + "\n"
	constexpr size_t kNormalSize = 3 * kComponentSize + 6; // "vn " + 3 components + "\n"

	const size_t nVertices = vertices.size();
	const size_t chunkVertices = (nVertices + maxThreads - 1) / maxThreads;
	char* vertexBlock = (char*)_aligned_malloc(kVertexSize * chunkVertices * maxThreads, 64);

	const size_t nTC = texcoords.size();
	const size_t chunkTC = (nTC + maxThreads - 1) / maxThreads;
	char* tcBlock = (char*)_aligned_malloc(kTexCoordSize * chunkTC * maxThreads, 64);

	const size_t nNormals = normals.size();
	const size_t chunkNormals = (nNormals + maxThreads - 1) / maxThreads;
	char* nrmBlock = (char*)_aligned_malloc(kNormalSize * chunkNormals * maxThreads, 64);

	std::vector<size_t> threadBufVertices(maxThreads);
	std::vector<size_t> threadBufTC(maxThreads);
	std::vector<size_t> threadBufNormals(maxThreads);

	std::string header;
	header.reserve(64);
	header.append("mtllib ");
	header.append(name);
	header.append(".mtl\n");

#ifndef _OPENMP
#error "OpenMP is NOT enabled in project settings!"
#endif

	// ------------------------------------------------------------
	// Vertices
	// ------------------------------------------------------------
	{
#pragma omp parallel num_threads(maxThreads)
		{
			const size_t tid = omp_get_thread_num();
			auto& localSize = threadBufVertices[tid];
			char* p = vertexBlock + tid * chunkVertices * kVertexSize;
			localSize = 0;

#pragma omp for schedule(static, chunkVertices)
			for (int64_t i = 0; i < static_cast<int64_t>(nVertices); ++i) {
				const auto& v = vertices[i];

				*p++ = 'v'; *p++ = ' ';
				p = WriteDouble(p, v[0]); *p++ = ' ';
				p = WriteDouble(p, v[1]); *p++ = ' ';
				p = WriteDouble(p, v[2]); *p++ = '\n';
			}

			localSize = (p - (vertexBlock + tid * chunkVertices * kVertexSize));
		}
	}

	// ------------------------------------------------------------
	// Texcoords
	// ------------------------------------------------------------
	if (nTC > 0) {
#pragma omp parallel num_threads(maxThreads)
		{
			const size_t tid = omp_get_thread_num();
			auto& localSize = threadBufTC[tid];
			char* p = tcBlock + tid * chunkTC * kTexCoordSize;
			localSize = 0;

#pragma omp for schedule(static, chunkTC)
			for (int64_t i = 0; i < static_cast<int64_t>(nTC); ++i) {
				const auto& tc = texcoords[i];

				*p++ = 'v'; *p++ = 't'; *p++ = ' ';
				p = WriteDouble(p, tc[0]); *p++ = ' ';
				p = WriteDouble(p, tc[1]); *p++ = '\n';
			}

			localSize = (p - (tcBlock + tid * chunkTC * kTexCoordSize));
		}
	}

	// ------------------------------------------------------------
	// Normals
	// ------------------------------------------------------------
	if (nNormals > 0) {
#pragma omp parallel num_threads(maxThreads)
		{
			const size_t tid = omp_get_thread_num();
			auto& localSize = threadBufNormals[tid];
			char* p = nrmBlock + tid * chunkNormals * kNormalSize;
			localSize = 0;

#pragma omp for schedule(static, chunkNormals)
			for (int64_t i = 0; i < static_cast<int64_t>(nNormals); ++i) {
				const auto& nrm = normals[i];

				*p++ = 'v'; *p++ = 'n'; *p++ = ' ';

				p = WriteDouble(p, nrm[0]); *p++ = ' ';
				p = WriteDouble(p, nrm[1]); *p++ = ' ';
				p = WriteDouble(p, nrm[2]); *p++ = '\n';
			}

			localSize = (p - (nrmBlock + tid * chunkNormals * kNormalSize));
		}
	}

	// ------------------------------------------------------------
	// Faces (single-threaded, deterministic)
	// ------------------------------------------------------------
	std::string facesBuf;
	facesBuf.reserve(groups.size() * 128);

	const bool hasTex = !texcoords.empty();
	const bool hasNorm = !normals.empty();

	for (const auto& group : groups) {
		facesBuf.append("usemtl ");
		facesBuf.append(group.material_name);
		facesBuf.push_back('\n');

		char line[128];
		for (const Face& face : group.faces) {
			char* p = line;

			*p++ = 'f';

			for (int k = 0; k < 3; ++k) {
				*p++ = ' ';
				p = WriteUInt(p, face.vertices[k] + OBJ_INDEX_OFFSET);

				if (hasTex || hasNorm) {
					*p++ = '/';
					if (hasTex)
						p = WriteUInt(p, face.texcoords[k] + OBJ_INDEX_OFFSET);
					if (hasNorm) {
						*p++ = '/';
						p = WriteUInt(p, face.normals[k] + OBJ_INDEX_OFFSET);
					}
				}
			}

			*p++ = '\n';
			facesBuf.append(line, p - line);
		}
	}

	// ------------------------------------------------------------
	// Final assembly (single allocation, single write)
	// ------------------------------------------------------------
	size_t totalSize = header.size() + facesBuf.size();
	for (const auto& s : threadBufVertices) totalSize += s;
	for (const auto& s : threadBufTC) totalSize += s;
	for (const auto& s : threadBufNormals) totalSize += s;

	std::string finalBuf;
	finalBuf.reserve(totalSize);

	finalBuf.append(header);
  for (int tid = 0; tid < maxThreads; ++tid) {
		finalBuf.append(vertexBlock + tid * chunkVertices * kVertexSize, threadBufVertices[tid]);
	}
  _aligned_free(vertexBlock);

	for (int tid = 0; tid < maxThreads; ++tid) {
		finalBuf.append(tcBlock + tid * chunkTC * kTexCoordSize, threadBufTC[tid]);
	}
	_aligned_free(tcBlock);

	for (int tid = 0; tid < maxThreads; ++tid) {
		finalBuf.append(nrmBlock + tid * chunkNormals * kNormalSize, threadBufNormals[tid]);
	}
	_aligned_free(nrmBlock);

	file.Write(finalBuf.data(), finalBuf.size());

	file.Write(facesBuf.data(), facesBuf.size());

	return true;
}
#else
bool ObjModel::Save(const String& fileName, unsigned precision, bool texLossless) const
{
	if (vertices.empty())
		return false;
	const String prefix(Util::getFileFullName(fileName));
	const String name(Util::getFileNameExt(prefix));

	if (!material_lib.Save(prefix, texLossless))
		return false;

	std::ofstream out((prefix + ".obj").c_str());
	if (!out.good())
		return false;

	out << "mtllib " << name << ".mtl" << "\n";

	out << std::fixed << std::setprecision(precision);
	for (size_t i = 0; i < vertices.size(); ++i) {
		out << "v "
			<< vertices[i][0] << " "
			<< vertices[i][1] << " "
			<< vertices[i][2] << "\n";
	}

	for (size_t i = 0; i < texcoords.size(); ++i) {
		out << "vt "
			<< texcoords[i][0] << " "
			<< texcoords[i][1] << "\n";
	}

	for (size_t i = 0; i < normals.size(); ++i) {
		out << "vn "
			<< normals[i][0] << " "
			<< normals[i][1] << " "
			<< normals[i][2] << "\n";
	}

	for (size_t i = 0; i < groups.size(); ++i) {
		out << "usemtl " << groups[i].material_name << "\n";
		for (size_t j = 0; j < groups[i].faces.size(); ++j) {
			const Face& face =  groups[i].faces[j];
			out << "f";
			for (size_t k = 0; k < 3; ++k) {
				out << " " << face.vertices[k]  + OBJ_INDEX_OFFSET;
				if (!texcoords.empty()) {
					out << "/" << face.texcoords[k]  + OBJ_INDEX_OFFSET;
					if (!normals.empty())
						out << "/" << face.normals[k]  + OBJ_INDEX_OFFSET;
				} else
				if (!normals.empty())
					out << "//" << face.normals[k]  + OBJ_INDEX_OFFSET;
			}
			out << "\n";
		}
	}
	return true;
}
#endif

bool ObjModel::Load(const String& fileName)
{
	ASSERT(vertices.empty() && groups.empty() && material_lib.materials.empty());
	std::ifstream fin(fileName.c_str());
	String line, keyword;
	std::istringstream in;
	while (fin.good()) {
		std::getline(fin, line);
		if (line.empty() || line[0u] == '#')
			continue;
		in.str(line);
		in >> keyword;
		if (keyword == "v") {
			Vertex v;
			in >> v[0] >> v[1] >> v[2];
			if (in.fail())
				return false;
			vertices.push_back(v);
		} else
		if (keyword == "vt") {
			TexCoord vt;
			in >> vt[0] >> vt[1];
			if (in.fail())
				return false;
			texcoords.push_back(vt);
		} else
		if (keyword == "vn") {
			Normal vn;
			in >> vn[0] >> vn[1] >> vn[2];
			if (in.fail())
				return false;
			normals.push_back(vn);
		} else
		if (keyword == "f") {
			Face f;
			memset(&f, 0xFF, sizeof(Face));
			for (size_t k = 0; k < 3; ++k) {
				in >> keyword;
				switch (sscanf(keyword, "%u/%u/%u", f.vertices+k, f.texcoords+k, f.normals+k)) {
				case 1:
					f.vertices[k] -= OBJ_INDEX_OFFSET;
					break;
				case 2:
					f.vertices[k] -= OBJ_INDEX_OFFSET;
					if (f.texcoords[k] != NO_ID)
						f.texcoords[k] -= OBJ_INDEX_OFFSET;
					if (f.normals[k] != NO_ID)
						f.normals[k] -= OBJ_INDEX_OFFSET;
					break;
				case 3:
					f.vertices[k] -= OBJ_INDEX_OFFSET;
					f.texcoords[k] -= OBJ_INDEX_OFFSET;
					f.normals[k] -= OBJ_INDEX_OFFSET;
					break;
				default:
					return false;
				}
			}
			if (in.fail())
				return false;
			if (groups.empty())
				AddGroup("");
			groups.back().faces.push_back(f);
		} else
		if (keyword == "mtllib") {
			in >> keyword;
			if (!material_lib.Load(keyword))
				return false;
		} else
		if (keyword == "usemtl") {
			Group group;
			in >> group.material_name;
			if (in.fail())
				return false;
			groups.push_back(group);
		}
		in.clear();
	}
	return !vertices.empty();
}


ObjModel::Group& ObjModel::AddGroup(const String& material_name)
{
	groups.push_back(Group());
	Group& group = groups.back();
	group.material_name = material_name;
	if (!GetMaterial(material_name))
		material_lib.materials.push_back(MaterialLib::Material(material_name));
	return group;
}

ObjModel::MaterialLib::Material* ObjModel::GetMaterial(const String& name)
{
	MaterialLib::Materials::iterator it(std::find_if(material_lib.materials.begin(), material_lib.materials.end(), [&name](const MaterialLib::Material& mat) { return mat.name == name; }));
	if (it == material_lib.materials.end())
		return NULL;
	return &(*it);
}
/*----------------------------------------------------------------*/
