/*
PoissonReconLib.cpp

In-process screened-Poisson reconstruction. This is the no-aux-data (position +
normal, optional per-vertex density) path of PoissonRecon.cpp's Execute(),
instantiated for the exact configuration the stock PoissonRecon.exe uses by
default (Real=float, Dim=3, degree=DefaultFEMDegree, boundary=DefaultFEMBoundary)
and driven from in-memory arrays / collected into in-memory vectors instead of
.ply files. All solver/extraction parameters are set to the same values the exe's
command line produces when only --depth/--samplesPerNode/--pointWeight/--density
are supplied (which is exactly how OpenMVS invokes it).
*/
#include "PreProcessor.h"
#include "Reconstructors.h"

#include <mutex>
#include <stdexcept>
#include <iostream>
#include <streambuf>
#include <string>
#include <cstring>
#include <cstdlib>
#if !defined( _WIN32 ) && !defined( _WIN64 )
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "MyMiscellany.h"
#include "FEMTree.h"
#include "VertexFactory.h"
#include "DataStream.imp.h"
#include "PoissonReconLib.h"

namespace
{
	using namespace PoissonRecon;

	typedef float Real;
	static const unsigned int Dim = 3;

	// Exactly the instantiation the stock exe uses (no FAST_COMPILE, default
	// degree/boundary): degree-1, Neumann boundary.
	static const unsigned int FEMSig = FEMDegreeAndBType< Reconstructor::Poisson::DefaultFEMDegree , Reconstructor::Poisson::DefaultFEMBoundary >::Signature;
	typedef IsotropicUIntPack< Dim , FEMSig > Sigs;
	typedef Reconstructor::Implicit< Real , Dim , Sigs > ImplicitType;
	typedef Reconstructor::Poisson::Solver< Real , Dim , Sigs > SolverType;

	// Input oriented-point stream backed directly by OpenMVS's contiguous arrays.
	// Read single-threaded (exactly as PoissonRecon reads its input PLY), so the
	// sample order — and thus the result — matches the exe.
	struct ArrayOrientedSampleStream : public Reconstructor::InputOrientedSampleStream< Real , Dim >
	{
		const float *pts , *nrm;
		size_t count , idx;
		ArrayOrientedSampleStream( const float *p , const float *n , size_t c ) : pts(p) , nrm(n) , count(c) , idx(0) {}

		void reset( void ){ idx = 0; }

		bool read( Reconstructor::Position< Real , Dim > &p , Reconstructor::Normal< Real , Dim > &n )
		{
			if( idx>=count ) return false;
			const float *pp = pts + idx*3 , *nn = nrm + idx*3;
			for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = (Real)pp[d] , n[d] = (Real)nn[d];
			idx++;
			return true;
		}
		bool read( unsigned int , Reconstructor::Position< Real , Dim > &p , Reconstructor::Normal< Real , Dim > &n ){ return read( p , n ); }
	};

	// Per-thread accumulation slot, padded to a cache line.
	//
	// A std::vector header is 24 B on MSVC, so a plain std::vector<std::vector<T>>
	// packs ~2.6 threads' headers into every 64-byte cache line -- and push_back
	// writes that header (the size pointer) on EVERY append. The extractor runs
	// ThreadPool::NumThreads() workers appending concurrently: 1.29M vertices and
	// 2.58M triangles on the measured run, so those shared lines ping-pong for the
	// whole of the level-set extraction (the 4.7 s "Got Faces" phase).
	//
	// alignas(64) is sufficient by itself -- it forces sizeof up to a multiple of
	// the alignment, so the slots land exactly one per line with no explicit
	// padding member. The array costs NumThreads() * 64 B. Over-aligned types get
	// the aligned operator new automatically under C++17, which this is built as.
	template< typename T >
	// n: per-thread write tally, living in the SAME cache line as the buffer so a
	// stream that does not need a globally-ordered index can avoid a shared atomic
	// entirely (see MemFaceStream::size).
	struct alignas(64) ThreadBuf { std::vector< T > v; size_t n = 0; };

	// Output vertex stream: append { x, y, z, density } per vertex. Writes arrive
	// concurrently from the parallel level-set extractor, so rather than serialize
	// every append through a mutex (lock contention + reallocation while holding the
	// lock), each worker thread accumulates into its own buffer (lock-free). A global
	// atomic counter assigns each vertex its final index at write time, so the face
	// stream's references stay valid; finalize() then scatters the per-thread buffers
	// into the contiguous output array. finalize() must be called once, after extraction.
	struct MemVertexStream : public Reconstructor::OutputLevelSetVertexStream< Real , Dim >
	{
		// One trivially-copyable record per vertex so the hot write() path is a
		// single push_back (MSVC inlines the fast path of a POD push_back but not
		// the four-plus separate scalar push_backs it replaces).
		struct VertRec { float x , y , z , w ; size_t idx ; };
		std::vector< float > &verts;
		bool density;
		std::atomic< size_t > count;
		std::vector< ThreadBuf< VertRec > > tData;   // per-thread vertex records, one per cache line
		MemVertexStream( std::vector< float > &v , bool d ) : verts(v) , density(d) , count(0) , tData( ThreadPool::NumThreads() ) {}

		size_t size( void ) const { return count.load( std::memory_order_relaxed ); }

		size_t write( unsigned int thread , const Reconstructor::Position< Real , Dim > &p , const Reconstructor::Gradient< Real , Dim > & , const Reconstructor::Weight< Real > &w )
		{
			const size_t i = count.fetch_add( 1 , std::memory_order_relaxed );
			tData[thread].v.push_back( VertRec{ (float)p[0] , (float)p[1] , (float)p[2] , density ? (float)w : 0.f , i } );
			return i;
		}
		size_t write( const Reconstructor::Position< Real , Dim > &p , const Reconstructor::Gradient< Real , Dim > &g , const Reconstructor::Weight< Real > &w ){ return write( 0u , p , g , w ); }

		// Scatter the per-thread buffers into the contiguous output. Each global index
		// was produced by exactly one thread, so the targets are disjoint and the
		// scatter parallelizes safely.
		void finalize( void )
		{
			verts.resize( count.load() * 4 );
			ThreadPool::ParallelFor( 0 , tData.size() , [&]( unsigned int , size_t t )
			{
				std::vector< VertRec > &d = tData[t].v;
				for( size_t k=0 ; k<d.size() ; k++ )
				{
					const VertRec &r = d[k];
					float *dst = &verts[ r.idx*4 ];
					dst[0] = r.x , dst[1] = r.y , dst[2] = r.z , dst[3] = r.w;
				}
				// Release this thread's buffer as soon as it has been scattered:
				// VertRec is 24 B/vertex against 16 B/vertex in the output, so
				// holding every buffer until the whole scatter finished cost
				// ~40 B/vertex at peak instead of ~16 B + one thread's share.
				// Each index t is visited by exactly one task, so mutating
				// tData[t] here is race-free. swap-with-empty, not clear():
				// clear() keeps the capacity allocated and frees nothing.
				std::vector< VertRec >().swap( d );
			} );
		}
	};

	// Output face stream: triangles (3 indices). Polygons (size>3) are fan-
	// triangulated as a safety fallback; with forceManifold + !polygonMesh the
	// extractor already emits triangles. Like the vertex stream, writes arrive
	// concurrently, so each thread accumulates triangles into its own buffer and
	// finalize() concatenates them (face order is irrelevant to the mesh; the face
	// indices already reference the correct global vertex indices).
	struct MemFaceStream : public Reconstructor::OutputFaceStream< Dim-1 >
	{
		// One trivially-copyable triangle per push_back (see MemVertexStream): a
		// single POD push_back inlines its fast path where three scalar pushes did not.
		struct Tri { uint32_t a , b , c ; };
		std::vector< uint32_t > &tris;
		std::vector< ThreadBuf< Tri > > tTris;  // per-thread triangles, one per cache line
		MemFaceStream( std::vector< uint32_t > &t ) : tris(t) , tTris( ThreadPool::NumThreads() ) {}

		// NO SHARED ATOMIC. This used to be a std::atomic<size_t> fetch_add per face --
		// 13.77M contended increments on ONE cache line across 32 threads on the measured
		// run -- and nothing needed the global ordinal it produced:
		//   * finalize() places triangles by prefix-summing the per-thread BUFFER SIZES,
		//     not by this counter;
		//   * all three polygonStream.write() call sites in FEMTree.LevelSet.3D.inl
		//     (1963, 1992, 2006) DISCARD the return value.
		// so the only consumer was size(), for the "Vertices / Faces" trace line.
		//
		// The vertex stream above is NOT the same case and keeps its atomic: its write()
		// return IS the vertex's global index, stored in VertRec and used for placement.
		//
		// Counts FACES (polygons), as before -- not triangles. They differ only if the
		// extractor emits polygons, which fan-triangulate below.
		size_t size( void ) const
		{
			size_t n = 0;
			for( size_t t=0 ; t<tTris.size() ; t++ ) n += tTris[t].n;
			return n;
		}

		size_t write( unsigned int thread , const Reconstructor::Face< Dim-1 > &f )
		{
			// Per-thread ordinal now, not a global one. Safe because every caller
			// discards it; see the note above before relying on this value.
			const size_t i = tTris[thread].n++;
			std::vector< Tri > &t = tTris[thread].v;
			if( f.size()==3 ) t.push_back( Tri{ (uint32_t)f[0] , (uint32_t)f[1] , (uint32_t)f[2] } );
			else for( size_t k=2 ; k<f.size() ; k++ ) t.push_back( Tri{ (uint32_t)f[0] , (uint32_t)f[k-1] , (uint32_t)f[k] } );
			return i;
		}
		size_t write( const Reconstructor::Face< Dim-1 > &f ){ return write( 0u , f ); }

		// Concatenate the per-thread triangle buffers into the output.
		//
		// Tri is three consecutive uint32_t with no padding (asserted below), so a
		// thread's buffer is ALREADY bit-identical to the flat (a,b,c,a,b,c,...) layout
		// of the output. Prefix-sum the per-thread counts, size the output once, then
		// memcpy each thread's block straight into its slot -- in parallel, since the
		// destination ranges are disjoint by construction.
		//
		// WAS: three push_backs per triangle on a single thread. That is 7.75M calls on
		// the measured run, each re-checking capacity, while the vertex stream directly
		// above it already scattered in parallel. Concatenating in thread order is what
		// the sequential loop did, so the face ORDER is unchanged.
		void finalize( void )
		{
			static_assert( sizeof( Tri )==3*sizeof( uint32_t ) , "Tri must be tightly packed for the memcpy path" );
			const size_t nT = tTris.size();
			std::vector< size_t > off( nT+1 , 0 );
			for( size_t t=0 ; t<nT ; t++ ) off[t+1] = off[t] + tTris[t].v.size();
			const size_t base = tris.size();
			tris.resize( base + off[nT]*3 );
			uint32_t * const dst = tris.data() + base;
			ThreadPool::ParallelFor( 0 , nT , [&]( unsigned int , size_t t )
			{
				// Release each per-thread buffer as soon as it has been copied (Tri is
				// 12 B/triangle against 12 B/triangle in the output, so holding them all
				// cost ~2x the triangle array at peak). Each index t is visited by exactly
				// one task, so mutating tTris[t] here is race-free. swap-with-empty, not
				// clear(): clear() keeps the capacity and frees nothing.
				std::vector< Tri > &s = tTris[t].v;
				if( !s.empty() ) memcpy( dst + off[t]*3 , s.data() , s.size()*sizeof( Tri ) );
				std::vector< Tri >().swap( s );
			} );
		}
	};

	// Available PHYSICAL memory in bytes; 0 when it cannot be determined (in which case
	// callers treat the budget as unlimited, i.e. exactly the pre-gate behaviour).
	//
	// Deliberately "available" rather than "total": the box this runs on is shared with
	// the host application and whatever else the user has open -- the [MESH-MEM] line in
	// ReconstructMesh routinely reports 40% of physical already held by other processes --
	// so sizing against total would budget memory that does not exist.
	static size_t AvailablePhysicalBytes( void )
	{
#if defined( _WIN32 ) || defined( _WIN64 )
		MEMORYSTATUSEX ms;
		ms.dwLength = sizeof( ms );
		if( GlobalMemoryStatusEx( &ms ) ) return (size_t)ms.ullAvailPhys;
		return 0;
#else
		const long pages = sysconf( _SC_AVPHYS_PAGES ) , pageSize = sysconf( _SC_PAGE_SIZE );
		if( pages>0 && pageSize>0 ) return (size_t)pages * (size_t)pageSize;
		return 0;
#endif
	}

	// Line-splitting streambuf. Accumulates characters and hands each COMPLETE line
	// to a caller-supplied sink; see ReconParams::logSink for why the solver's
	// std::cout trace has to be intercepted rather than simply read.
	//
	// Splitting on '\n' rather than forwarding raw writes is the whole point: the
	// host's logger is line-oriented (one call = one timestamped record), while a
	// single "# Linear system solved: ..." line reaches a streambuf as several
	// separate writes -- the literal, then the profiler's fields, then std::endl.
	// Forwarding those verbatim would produce four fragmentary log records per phase.
	struct LineSinkBuf : public std::streambuf
	{
		explicit LineSinkBuf( void (*sink)( const char* ) ) : _sink(sink) { _line.reserve( 256 ); }
		~LineSinkBuf( void ) { _Flush(); }

	protected:
		// Single-character path. std::endl and the ostream fallbacks land here.
		int_type overflow( int_type c=traits_type::eof() ) override
		{
			if( c==traits_type::eof() ) return traits_type::not_eof( c );
			std::lock_guard< std::mutex > lock( _mtx );
			_Put( traits_type::to_char_type( c ) );
			return c;
		}
		// Bulk path. REQUIRED, not an optimization: ostream::operator<<(const char*)
		// and operator<<(std::string) both go through sputn, so without this every
		// phase line would be re-entered through overflow() one character at a time.
		std::streamsize xsputn( const char *s , std::streamsize n ) override
		{
			std::lock_guard< std::mutex > lock( _mtx );
			for( std::streamsize i=0 ; i<n ; i++ ) _Put( s[i] );
			return n;
		}
		// std::endl flushes; so does the redirect guard before it restores std::cout,
		// which is what keeps an unterminated tail from being lost.
		int sync( void ) override
		{
			std::lock_guard< std::mutex > lock( _mtx );
			_Flush();
			return 0;
		}

	private:
		// Both unlocked -- every caller above already holds _mtx (the destructor runs
		// after the redirect is gone, so it is single-threaded by then).
		void _Flush( void )
		{
			// Trailing '\r' and padding: the trace right-aligns its phase labels, and a
			// blank line is the solver's own paragraph break, not a record worth keeping.
			while( !_line.empty() && ( _line.back()=='\r' || _line.back()==' ' || _line.back()=='\t' ) ) _line.pop_back();
			if( !_line.empty() && _sink ) _sink( _line.c_str() );
			_line.clear();
		}
		void _Put( char ch )
		{
			if( ch=='\n' ) { _Flush(); return; }
			// A phase line is ~80 characters. The bound exists only so that output which
			// never terminates a line cannot grow the buffer without limit.
			if( _line.size()<4096 ) _line.push_back( ch );
		}

		void (*_sink)( const char* );
		std::string _line;
		std::mutex  _mtx;   // the level-set extractor is threaded and can print
	};

	// Swap std::cout's streambuf for the duration of a scope and ALWAYS put the
	// original back -- including on the exception path, which the solve has.
	// Flushes before restoring so a partial last line still reaches the sink.
	struct CoutRedirect
	{
		CoutRedirect( std::streambuf *nb , bool active ) : _old(NULL) , _active(active)
		{
			if( _active ) _old = std::cout.rdbuf( nb );
		}
		~CoutRedirect( void )
		{
			if( !_active ) return;
			std::cout.flush();
			std::cout.rdbuf( _old );
		}
		CoutRedirect( const CoutRedirect& ) = delete;
		CoutRedirect &operator=( const CoutRedirect& ) = delete;
	private:
		std::streambuf *_old;
		bool _active;
	};
}

namespace PoissonReconLib
{
	bool Reconstruct( const float *points , const float *normals , size_t numPoints , const ReconParams &params , Mesh &out )
	{
		out.Clear();
		if( !points || !normals || numPoints==0 ) return false;

		// Threading defaults, matching the exe's main() (which copies the CLI
		// defaults: parallel type 0, default schedule/chunk size).
		ThreadPool::ParallelizationType = (ThreadPool::ParallelType)0;

		// Worker count. PoissonRecon's pool is independent of OpenMVS's and of
		// OMP_NUM_THREADS (ParallelFor passes num_threads() explicitly), so unless we
		// set it here the solve runs on every core no matter what --max-threads said.
		// Precedence: OPENMVS_POISSON_THREADS env var > params.threads > the process
		// OpenMP setting. Set before anything sizes per-thread scratch from
		// ThreadPool::NumThreads() -- see the warning on SetNumThreads.
		{
			unsigned int nThreads = params.threads;
			if( const char *e = std::getenv( "OPENMVS_POISSON_THREADS" ) )
			{
				const int v = std::atoi( e );
				if( v>0 ) nThreads = (unsigned int)v;
			}
#ifdef _OPENMP
			if( !nThreads )
			{
				const int m = omp_get_max_threads();
				if( m>0 ) nThreads = (unsigned int)m;
			}
#endif
			ThreadPool::SetNumThreads( nThreads );
		}

		typename Reconstructor::Poisson::SolutionParameters< Real > sParams;
		sParams.verbose                = params.verbose;
		sParams.dirichletErode         = true;     // --noErode not set
		sParams.exactInterpolation     = false;    // --exact not set
		sParams.showResidual           = false;
		sParams.confidence             = false;    // --confidence not set
		sParams.scale                  = (Real)params.scale;
		sParams.lowDepthCutOff         = (Real)0.;
		sParams.width                  = (Real)0.;
		sParams.pointWeight            = (Real)params.pointWeight;
		sParams.samplesPerNode         = (Real)params.samplesPerNode;
		sParams.cgSolverAccuracy       = (Real)1e-3;
		sParams.perLevelDataScaleFactor= (Real)32.;
		sParams.depth                  = (unsigned int)params.depth;
		sParams.baseDepth              = (unsigned int)(-1); // CLI default -1 (auto)
		sParams.solveDepth             = (unsigned int)(-1);
		sParams.fullDepth              = 5;
		sParams.kernelDepth            = (unsigned int)(-1);
		sParams.envelopeDepth          = (unsigned int)(-1);
		sParams.baseVCycles            = 1;
		sParams.iters                  = 8;
		sParams.alignDir               = Dim-1;


		Reconstructor::LevelSetExtractionParameters meParams;
		meParams.linearFit         = false;   // --linearFit not set
		meParams.outputGradients   = false;   // --gradients not set
		meParams.forceManifold     = true;    // --nonManifold not set
		meParams.polygonMesh       = false;   // --polygonMesh not set
		meParams.gridCoordinates   = false;
		meParams.outputDensity     = params.density;
		meParams.verbose           = params.verbose;

		// Capture the solver's own per-phase trace into the host's logger. Declared
		// OUTSIDE the try so it is still installed while the catch handler runs and
		// while `implicit` is destroyed, and so its destructor restores std::cout on
		// every exit path -- normal return, early return, or throw.
		LineSinkBuf  logBuf( params.logSink );
		CoutRedirect logRedirect( &logBuf , params.verbose && params.logSink!=NULL );

		// First line through the sink: without it there is no way to tell from a log
		// whether a thread-count override actually took effect.
		if( params.verbose )
			std::cout << "Threads: " << ThreadPool::NumThreads() << std::endl;


		ImplicitType *implicit = NULL;
		try
		{
			// Extent, in parallel, from the array we already hold -- so Solve() does not
			// stream all numPoints a second time just to find the bounding box.
			//
			// Solve() normally calls PointExtent::GetXForm on the sample stream before it
			// inserts anything, which walks every point through a virtual read() on one
			// thread. On the larger measured scene that is 53.1M points, and it is the
			// first half of a phase that reports 3.7 s.
			//
			// BIT-IDENTICAL, not equivalent-within-tolerance. Extent is min/max over a
			// fixed 9-direction frame; min and max are associative, commutative and exact
			// in floating point, so per-thread partials merged with the Extent::operator+
			// that already exists give exactly the serial result regardless of how the
			// points are divided. The transform is then produced by the SAME
			// GetXForm(extent,scale,dir) overload the streaming version ends in, with the
			// same scale and alignDir, so nothing about the octree cube changes.
			//
			// Reads only positions. The streaming version also pulls the normal through
			// its read() and discards it; that is the other half of what this avoids.
			XForm< Real , Dim+1 > modelToUnitCube;
			bool haveModelToUnitCube = false;
			if( sParams.scale>0 && numPoints )
			{
				const double _extentT = Time();
				typedef PointExtent::Extent< Real , Dim , true > ExtentType;
				std::vector< ExtentType > tExtents( ThreadPool::NumThreads() );
				ThreadPool::ParallelFor( 0 , numPoints , [&]( unsigned int t , size_t i )
					{
						// Element-wise, exactly as ArrayOrientedSampleStream::read builds it,
						// so the values fed to add() are the same objects the serial pass saw.
						const float *pp = points + i*3;
						Point< Real , Dim > p;
						for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = (Real)pp[d];
						tExtents[t].add( p );
					} );
				ExtentType extent;
				for( size_t t=0 ; t<tExtents.size() ; t++ ) extent = extent + tExtents[t];
				modelToUnitCube = PointExtent::GetXForm< Real , Dim , true >( extent , (Real)sParams.scale , sParams.alignDir );
				haveModelToUnitCube = true;
				// Report what this pass cost, because it happens BEFORE Solve() and therefore
				// outside the "# Read input into tree" profiler window that used to contain the
				// serial extent walk it replaces. Without this line the phase looks cheaper by
				// however long this takes, and there is no way to tell a real saving from work
				// that merely moved somewhere untimed.
				//
				// It matters: the serial walk's removal cut that phase 3.7 -> 2.7 s on Historic
				// (53.1M points) and by NOTHING measurable on SchnellTests (28.2M), where half
				// the saving was expected. One of those two numbers is not what it appears, and
				// this line is what will say which.
				if( params.verbose )
					std::cout << "Parallel extent: " << ( Time()-_extentT ) << " (s) over " << numPoints << " points ("
					          << ThreadPool::NumThreads() << " threads) -- NOT included in 'Read input into tree'" << std::endl;
			}

			ArrayOrientedSampleStream sampleStream( points , normals , numPoints );
			implicit = SolverType::Solve( sampleStream , sParams ,
				(const typename Reconstructor::Poisson::EnvelopeMesh< Real , Dim >*)NULL ,
				(Reconstructor::InputValuedSampleStream< Real , Dim >*)NULL ,
				haveModelToUnitCube ? &modelToUnitCube : NULL );
			if( !implicit ) return false;

			MemVertexStream vStream( out.vertices , params.density );
			MemFaceStream   fStream( out.triangles );
			implicit->extractLevelSet( vStream , fStream , meParams );
			// Flush the per-thread buffers into the contiguous output arrays.
			vStream.finalize();
			fStream.finalize();
		}
		catch( const std::exception &e )
		{
			if( implicit ) delete implicit;
			MK_WARN( "PoissonReconLib::Reconstruct failed: " , e.what() );
			out.Clear();
			return false;
		}
		delete implicit;
		return out.VertexCount()>0 && out.TriangleCount()>0;
	}
}
