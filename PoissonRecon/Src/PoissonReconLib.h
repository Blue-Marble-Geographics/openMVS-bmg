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
	};

	// Parameters for the surface trimmer (SurfaceTrimmer.exe equivalents).
	struct TrimParams
	{
		float trim          = 7.0f;    // --trim   (density threshold)
		float aRatio        = 0.002f;  // --aRatio (island area ratio)
		bool  removeIslands = true;    // --removeIslands
		bool  verbose       = false;
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
