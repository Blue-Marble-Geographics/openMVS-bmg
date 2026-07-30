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
		std::vector< std::vector< VertRec > > tData;   // per-thread vertex records
		MemVertexStream( std::vector< float > &v , bool d ) : verts(v) , density(d) , count(0) , tData( ThreadPool::NumThreads() ) {}

		size_t size( void ) const { return count.load( std::memory_order_relaxed ); }

		size_t write( unsigned int thread , const Reconstructor::Position< Real , Dim > &p , const Reconstructor::Gradient< Real , Dim > & , const Reconstructor::Weight< Real > &w )
		{
			const size_t i = count.fetch_add( 1 , std::memory_order_relaxed );
			tData[thread].push_back( VertRec{ (float)p[0] , (float)p[1] , (float)p[2] , density ? (float)w : 0.f , i } );
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
				const std::vector< VertRec > &d = tData[t];
				for( size_t k=0 ; k<d.size() ; k++ )
				{
					const VertRec &r = d[k];
					float *dst = &verts[ r.idx*4 ];
					dst[0] = r.x , dst[1] = r.y , dst[2] = r.z , dst[3] = r.w;
				}
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
		std::atomic< size_t > count;
		std::vector< std::vector< Tri > > tTris;  // per-thread triangles
		MemFaceStream( std::vector< uint32_t > &t ) : tris(t) , count(0) , tTris( ThreadPool::NumThreads() ) {}

		size_t size( void ) const { return count.load( std::memory_order_relaxed ); }

		size_t write( unsigned int thread , const Reconstructor::Face< Dim-1 > &f )
		{
			const size_t i = count.fetch_add( 1 , std::memory_order_relaxed );
			std::vector< Tri > &t = tTris[thread];
			if( f.size()==3 ) t.push_back( Tri{ (uint32_t)f[0] , (uint32_t)f[1] , (uint32_t)f[2] } );
			else for( size_t k=2 ; k<f.size() ; k++ ) t.push_back( Tri{ (uint32_t)f[0] , (uint32_t)f[k-1] , (uint32_t)f[k] } );
			return i;
		}
		size_t write( const Reconstructor::Face< Dim-1 > &f ){ return write( 0u , f ); }

		// Concatenate the per-thread triangle buffers into the output (sequential).
		void finalize( void )
		{
			size_t total = 0;
			for( const std::vector< Tri > &t : tTris ) total += t.size();
			tris.reserve( tris.size() + total*3 );
			for( const std::vector< Tri > &t : tTris ) for( const Tri &tr : t ) tris.push_back( tr.a ) , tris.push_back( tr.b ) , tris.push_back( tr.c );
		}
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

		typename Reconstructor::Poisson::SolutionParameters< Real > sParams;
		sParams.verbose                = params.verbose;
		sParams.dirichletErode         = true;     // --noErode not set
		sParams.exactInterpolation     = false;    // --exact not set
		sParams.showResidual           = false;
		sParams.confidence             = false;    // --confidence not set
		sParams.scale                  = (Real)1.1;
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

		ImplicitType *implicit = NULL;
		try
		{
			ArrayOrientedSampleStream sampleStream( points , normals , numPoints );
			implicit = SolverType::Solve( sampleStream , sParams , (const typename Reconstructor::Poisson::EnvelopeMesh< Real , Dim >*)NULL );
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
