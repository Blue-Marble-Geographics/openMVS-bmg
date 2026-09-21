/*
PoissonReconLib.h

A thin, plain-C++ (no-template, no-PLY) wrapper around Kazhdan's PoissonRecon +
SurfaceTrimmer so they can be called in-process from OpenMVS instead of shelling
out to PoissonRecon.exe / SurfaceTrimmer.exe with temporary .ply files.

Only this header is included by the OpenMVS side; all of the heavily-templated
PoissonRecon machinery is confined to PoissonReconLib.cpp / SurfaceTrimmerLib.cpp,
which are compiled into a small static library. The reconstruction/trimming math
is byte-for-byte the same code the executables run (same FEM degree/boundary,
same trim pipeline) — only the I/O path changes from disk to memory.
*/
#ifndef POISSON_RECON_LIB_INCLUDED
#define POISSON_RECON_LIB_INCLUDED

#include <cstddef>
#include <cstdint>
#include <vector>

namespace PoissonReconLib
{
	// A reconstructed/trimmed mesh in a flat, ABI-stable layout.
	//   vertices : 4 floats per vertex = { x, y, z, density }
	//              (density is PoissonRecon's per-vertex "value"/weight, used by
	//               the trimmer's --trim threshold)
	//   triangles: 3 indices per triangle into the vertex array
	struct Mesh
	{
		std::vector< float >    vertices;  // size = 4 * vertexCount
		std::vector< uint32_t > triangles; // size = 3 * triangleCount

		size_t VertexCount  ( void ) const { return vertices.size() / 4; }
		size_t TriangleCount( void ) const { return triangles.size() / 3; }
		void   Clear        ( void ) { vertices.clear(); triangles.clear(); }
	};

	// Parameters for the screened-Poisson reconstruction. Mirrors the subset of
	// PoissonRecon.exe flags OpenMVS uses; everything else takes the exe defaults.
	struct ReconParams
	{
		int   depth          = 11;     // --depth
		float samplesPerNode = 1.5f;   // --samplesPerNode
		float pointWeight    = 4.0f;   // --pointWeight
		bool  density        = true;   // --density (emit per-vertex density)
		bool  verbose        = false;  // --verbose
		// --scale: the octree cube is scale * (max axis extent of the input points),
		// so cell = scale * extent / 2^depth. Was hardcoded 1.1 (the CLI default).
		// Exposed because the octree cube is a FREE parameter and the achievable cell
		// is otherwise quantised in factors of 2: a scene whose cell at depth D lands
		// just under the quality floor has to fall back to D-1 and a cell twice as
		// coarse as it needed. Padding the cube instead lets depth D land the cell
		// exactly on the floor. See POISSON_CUBE_PAD_MAX in SceneReconstruct.cpp.
		float scale          = 1.1f;   // --scale

		// Worker threads for the solve. 0 = inherit the process-wide OpenMP setting
		// (omp_get_max_threads(), which ReconstructMesh's --max-threads already sets),
		// which is what a caller normally wants.
		//
		// This exists because PoissonRecon keeps its OWN thread pool, entirely separate
		// from OpenMVS's: ThreadPool::Init is never called anywhere in the tree and
		// ThreadPool::_NumThreads defaults to std::thread::hardware_concurrency(), while
		// ParallelFor passes num_threads(_NumThreads) explicitly -- which also overrides
		// the OMP_NUM_THREADS environment variable. So before this field the solve
		// ignored --max-threads completely and always ran on every core; measured on
		// RichmondHistoric, --max-threads 1 left every Poisson phase unchanged (normal
		// field 17.8s -> 17.5s) while OpenMVS's own stages duly slowed down.
		//
		// The env var OPENMVS_POISSON_THREADS overrides this field when set, so the
		// thread count can be swept without a rebuild.
		unsigned int threads = 0;

		// Sink for the solver's OWN per-phase trace -- "# Read input into tree",
		// "#   Got kernel density", "#     Got normal field", "#       Finalized tree",
		// "# Linear system solved", "#            Got Faces", the node/memory tallies.
		// Each line carries that phase's elapsed time, so together they are the only
		// account of where the solve's wall clock actually goes.
		//
		// PoissonRecon writes them to std::cout, and the OpenMVS logger reads nothing
		// from std::cout: LogConsole swaps in a streambuf only when AllocConsole()
		// SUCCEEDED (i.e. the process had no console), and that streambuf fputc's to
		// stdout rather than raising a log record. So on a normal run the trace was
		// generated in full and then dropped on the floor, leaving the single most
		// expensive stage in the mesh pipeline as one unexplained interval in the log.
		//
		// When this is set, Reconstruct() swaps std::cout's streambuf for the duration
		// of the solve and hands each COMPLETE line here instead. Only takes effect
		// when `verbose` is also set; NULL leaves std::cout exactly where it pointed.
		// Not reentrant -- it mutates process-global std::cout -- so do not run two
		// Reconstruct() calls concurrently with a sink installed.
		void (*logSink)( const char* line ) = nullptr;
	};

	// Parameters for the surface trimmer (SurfaceTrimmer.exe equivalents).
	struct TrimParams
	{
		float trim          = 7.0f;    // --trim   (density threshold)
		float aRatio        = 0.002f;  // --aRatio (island area ratio)
		bool  removeIslands = true;    // --removeIslands
		bool  verbose       = false;

		// Sink for the trimmer's per-phase trace, same contract as ReconParams::logSink.
		// Trim does NOT install the std::cout redirect that Reconstruct does, so without
		// this the [TRIM-PROFILE] line goes to stdout and never reaches the log -- trim is
		// the largest single item in POST-SOLVE (3.7-3.9 s) and was completely unattributed
		// because of exactly that.
		void (*logSink)( const char* line ) = nullptr;
	};

	// Screened-Poisson surface reconstruction, in memory.
	//   points  : numPoints*3 floats (x,y,z), contiguous
	//   normals : numPoints*3 floats (nx,ny,nz), contiguous (unit-length not required)
	//   out     : reconstructed mesh (vertices carry density when params.density)
	// Returns false on failure.
	bool Reconstruct
	(
		const float* points ,
		const float* normals ,
		size_t numPoints ,
		const ReconParams& params ,
		Mesh& out
	);

	// Density-based surface trimming + island removal, in memory. Operates on a
	// mesh carrying per-vertex density (as produced by Reconstruct with density on).
	// Returns false on failure.
	bool Trim
	(
		const Mesh& in ,
		const TrimParams& params ,
		Mesh& out
	);
}

#endif // POISSON_RECON_LIB_INCLUDED
