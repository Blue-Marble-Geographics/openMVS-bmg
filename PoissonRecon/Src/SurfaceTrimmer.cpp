/*
Copyright (c) 2013, Michael Kazhdan
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#include "PreProcessor.h"

#define DEFAULT_DIMENSION 3

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>
#include <unordered_set>
#include <list>
#include "FEMTree.h"
#include "MyMiscellany.h"
#include "CmdLineParser.h"
#include "MAT.h"
#include "Geometry.h"
#include "Ply.h"
#include "VertexFactory.h"
#include "PoissonReconLib.h"

using namespace PoissonRecon;

// [PROFILE-ONLY] Print per-phase timing of the trimmer so we can see where the
// time goes (Split / connected-components+graph build / island-merge /
// triangulate / cleanup). Prints unconditionally (no --verbose needed). Set to 0
// to silence once the hot phase is identified.
#ifndef PR_TRIM_PROFILE
#define PR_TRIM_PROFILE 0
#endif // PR_TRIM_PROFILE

// [DENSITY-BAND FILTER] Targeted removal of the low-confidence, down-facing
// Poisson "underside/skirt" faces that appear only when --trim is lowered to keep
// completeness. --trim thresholds the per-vertex Poisson DENSITY; the artifact
// faces live in an ABSOLUTE density band [LO,HI) (measured from the diagnostic
// histogram, NOT relative to trim -- so changing --trim never shifts the band into
// well-supported geometry). This pre-pass (in PoissonReconLib::Trim, before
// TrimMeshInMemory) drops faces whose MAX vertex density is in [LO,HI) AND that are
// NOT upward-facing (unit face-normal.z < PR_DENSITY_BAND_MIN_UP_X100/100). Faces
// with density >= HI (confident) and the up-facing genuine fringe inside the band
// are kept, so it never harms well-supported geometry -- unlike a global cull or a
// raised trim. ASSUMES +Z is up and Poisson output oriented outward (top surface
// nz~+1, underside nz<0). Compile-time gated; 0 = feature off (pure A/B switch).
//   LO/HI = absolute density band bounds (x100). Set from the histogram: the range
//           where the artifact lives (e.g. [6,8) -> LO=600, HI=800). Run --trim at
//           or below LO so plain trim keeps completeness and the band does the rest.
//   MIN_UP = keep a band face unless its unit normal.z is below this. 10 = drop
//            down + near-vertical drapes; NEGATIVE = drop only clearly-down faces.
#ifndef PR_DENSITY_BAND_FILTER
#define PR_DENSITY_BAND_FILTER 0
#endif // PR_DENSITY_BAND_FILTER
#ifndef PR_DENSITY_BAND_LO_X100
#define PR_DENSITY_BAND_LO_X100 600   // band low  bound = 6.00 density (absolute)
#endif // PR_DENSITY_BAND_LO_X100
#ifndef PR_DENSITY_BAND_HI_X100
#define PR_DENSITY_BAND_HI_X100 800   // band high bound = 8.00 density (absolute)
#endif // PR_DENSITY_BAND_HI_X100
#ifndef PR_DENSITY_BAND_MIN_UP_X100
#define PR_DENSITY_BAND_MIN_UP_X100 (-30)   // drop band face only if unit normal.z < -0.30 (clearly-down underside; spares vertical/sloped legit edges)
#endif // PR_DENSITY_BAND_MIN_UP_X100
// OCCLUSION gate: a band down-facing face is only an "underside" if it is BURIED,
// i.e. real mesh exists above it at the same XY. A genuine top-edge lip has nothing
// above it and must be KEPT. This is the discriminator that density+orientation
// alone cannot provide. Build a top-down max-Z height grid; drop a candidate only
// if (localTopZ - faceCentroidZ) > BURY_MULT * medianEdge. This is applied ONLY to
// band + down-facing candidates, so unlike a global keep-top cull it cannot touch
// interior/up-facing geometry. Set OCCLUSION 0 to disable this gate (revert to
// band+orientation only). ASSUMES +Z up.
#ifndef PR_DENSITY_BAND_OCCLUSION
#define PR_DENSITY_BAND_OCCLUSION 1   // 1 = require buried-under-higher-geometry before dropping
#endif // PR_DENSITY_BAND_OCCLUSION
#ifndef PR_DENSITY_BAND_CELL_MULT_X100
#define PR_DENSITY_BAND_CELL_MULT_X100 200   // top-Z grid cell = 2.00 x median edge
#endif // PR_DENSITY_BAND_CELL_MULT_X100
#ifndef PR_DENSITY_BAND_BURY_MULT_X100
#define PR_DENSITY_BAND_BURY_MULT_X100 800   // CONSERVATIVE: only cut faces hanging > 8.00 x median edge below local ground (extreme bottom outliers)
#endif // PR_DENSITY_BAND_BURY_MULT_X100
// Reference is the LOCAL GROUND = the LOWEST confident-surface top over a
// (2*RADIUS+1) cell neighborhood, not the same-cell max (which is canopy). This
// spares ground under canopy (it sits at local ground) while still catching
// lobes (they dip below local ground), and fills empty cells so lobes with no
// confident vertex directly above are still caught. Larger radius reaches open
// ground from deeper under canopy but risks under-removing lobes near real relief.
#ifndef PR_DENSITY_BAND_NBR_RADIUS
#define PR_DENSITY_BAND_NBR_RADIUS 2   // neighborhood radius in cells for the local-ground reference
#endif // PR_DENSITY_BAND_NBR_RADIUS

#ifndef POISSONRECON_LIB_BUILD
CmdLineParameter< char* >
	In( "in" ) ,
	Out( "out" );
CmdLineParameter< float >
	Trim( "trim" ) ,
	IslandAreaRatio( "aRatio" , 0.001f );
CmdLineReadable
	PolygonMesh( "polygonMesh" ) ,
	Long( "long" ) ,
	ASCII( "ascii" ) ,
	RemoveIslands( "removeIslands" ) ,
	Debug( "debug" ) ,
	Verbose( "verbose" );


CmdLineReadable* params[] =
{
	&In , &Out , &Trim , &PolygonMesh , &IslandAreaRatio , &Verbose , &Long , &ASCII , &RemoveIslands , &Debug ,
	NULL
};

void ShowUsage( char* ex )
{
	printf( "Usage: %s\n" , ex );
	printf( "\t --%s <input polygon mesh>\n" , In.name );
	printf( "\t --%s <trimming value>\n" , Trim.name );
	printf( "\t[--%s <ouput polygon mesh>]\n" , Out.name );
	printf( "\t[--%s <relative area of islands>=%f]\n" , IslandAreaRatio.name , IslandAreaRatio.value );
	printf( "\t[--%s]\n" , RemoveIslands.name );
	printf( "\t[--%s]\n" , Debug.name );
	printf( "\t[--%s]\n" , PolygonMesh.name );
	printf( "\t[--%s]\n" , Long.name );
	printf( "\t[--%s]\n" , ASCII.name );
	printf( "\t[--%s]\n" , Verbose.name );
}
#endif // POISSONRECON_LIB_BUILD

template< typename Index >
struct ComponentGraph
{
	struct Node
	{
		double area;
		std::vector< Node * > neighbors;
		// Each entry is one component's polygon-index list. Storing whole vectors
		// (rather than individual indices) keeps merge()'s splice O(1) while making
		// population a handful of bulk copies instead of millions of list-node allocs.
		std::list< std::vector< Index > > polygonIndices;

		Node( void ) : area(0) {}

		void merge( void )
		{
			auto PopBack =[&]( std::vector< Node * > &nodes , size_t idx )
			{
				nodes[idx] = nodes.back();
				nodes.pop_back();
			};

			if( !neighbors.size() ) MK_THROW( "No neighbors" );

			// Remove the node from the neighbors of the neighbors
			for( unsigned int i=0 ; i<neighbors.size() ; i++ ) for( int j=(int)neighbors[i]->neighbors.size()-1 ; j>=0 ; j-- ) if( neighbors[i]->neighbors[j]==this )
				PopBack( neighbors[i]->neighbors , j );

			// Merge the node into its first neighbor
			Node *first = neighbors[0];
			first->area += area;
			first->polygonIndices.splice( first->polygonIndices.end() , polygonIndices );

			// Merge the remaining neighbors into the first neighbor
			for( unsigned int i=1 ; i<neighbors.size() ; i++ )
			{
				first->area += neighbors[i]->area;
				first->polygonIndices.splice( first->polygonIndices.end() , neighbors[i]->polygonIndices );
				for( unsigned int j=0 ; j<neighbors[i]->neighbors.size() ; j++ )
				{
					bool foundNeighbor = false;
					for( int k=(int)neighbors[i]->neighbors[j]->neighbors.size()-1 ; k>=0 ; k-- )
						if( neighbors[i]->neighbors[j]->neighbors[k]==neighbors[i] ) PopBack( neighbors[i]->neighbors[j]->neighbors , k );

					for( unsigned int k=0 ; k<first->neighbors.size() ; k++ ) foundNeighbor |= neighbors[i]->neighbors[j]==first->neighbors[k];
					if( !foundNeighbor )
					{
						first->neighbors.push_back( neighbors[i]->neighbors[j] );
						neighbors[i]->neighbors[j]->neighbors.push_back( first );
					}
				}

				neighbors[i]->area = 0;
				neighbors[i]->neighbors.clear();
			}
			// Clean up the node
			polygonIndices.clear();
			area = 0;
		}
	};

	static void SanityCheck( size_t count , std::function< const Node * ( size_t ) > nodeFunction )
	{
		std::unordered_map< const Node * , int > flags;
		for( unsigned int i=0 ; i<count ; i++ ) flags[ nodeFunction(i) ] = 0;
		for( auto iter=flags.begin() ; iter!=flags.end() ; iter++ ) if( !iter->second ) _PropagateFlag( flags , iter->first , 1 );

		for( auto iter=flags.begin() ; iter!=flags.end() ; iter++ ) for( unsigned int j=0 ; j<iter->first->neighbors.size() ; j++ )
		{
			if( iter->second==flags[ iter->first->neighbors[j] ] ) MK_THROW( "Not a bipartite graph" );
			bool foundSelf = false;
			for( unsigned int k=0 ; k<iter->first->neighbors[j]->neighbors.size() ; k++ ) if( iter->first->neighbors[j]->neighbors[k]==iter->first ) foundSelf = true;
			if( !foundSelf ) MK_THROW( "Asymmetric graph" );
		}
	}

protected:
	static void _PropagateFlag( std::unordered_map< const Node * , int > &flags , const Node *n , int flag )
	{
		if( flags[n] ) return;
		else
		{
			flags[n] = flag;
			for( unsigned int i=0 ; i<n->neighbors.size() ; i++ ) _PropagateFlag( flags , n->neighbors[i] , -flag );
		}

	}

};

template< typename Real , unsigned int Dim , typename ... AuxData >
using ValuedPointData = DirectSum< Real , Point< Real , Dim > , Real , AuxData ... >;

template< typename Index >
size_t BoostHash( Index i1 , Index i2 )
{
	size_t hash = (size_t)i1 + 0x9e3779b9;
	hash ^= (size_t)i2 + 0x9e3779b9 + (hash<<6) + (hash>>2);
	return hash;
}

template< typename Index >
struct EdgeKey
{
	Index key1 , key2;
	EdgeKey( Index k1=0 , Index k2=0 )
	{
		if( k1<k2 ) key1 = k1 , key2 = k2;
		else        key1 = k2 , key2 = k1;
	}
	bool operator == ( const EdgeKey &key ) const  { return key1==key.key1 && key2==key.key2; }
	struct Hasher{ size_t operator()( const EdgeKey &key ) const { return BoostHash(key.key1,key.key2); } };
};

template< typename Index >
struct HalfEdgeKey
{
	Index key1 , key2;
	HalfEdgeKey( Index k1=0 , Index k2=0 ) : key1(k1) , key2(k2) {}
	HalfEdgeKey opposite( void ) const { return HalfEdgeKey( key2 , key1 ); }
	bool operator == ( const HalfEdgeKey &key ) const  { return key1==key.key1 && key2==key.key2; }
	struct Hasher{ size_t operator()( const HalfEdgeKey &key ) const { return BoostHash(key.key1,key.key2); } };
};

template< typename Real , unsigned int Dim ,  typename ... AuxData >
ValuedPointData< Real , Dim , AuxData ... > InterpolateVertices( const ValuedPointData< Real , Dim , AuxData ... >& v1 , const ValuedPointData< Real , Dim , AuxData ... >& v2 , Real value )
{
	if( v1.template get<1>()==v2.template get<1>() ) return (v1+v2)/Real(2.);
	Real dx = ( v1.template get<1>()-value ) / ( v1.template get<1>()-v2.template get<1>() );
	return v1 * (Real)(1.-dx) + v2*dx;
}

template< typename Real , unsigned int Dim , typename Index , typename ... AuxData >
void SplitPolygon
(
	const std::vector< Index >& polygon ,
	std::vector< ValuedPointData< Real , Dim , AuxData ... > >& vertices ,
	std::vector< std::vector< Index > >* ltPolygons , std::vector< std::vector< Index > >* gtPolygons ,
	std::vector< bool >* ltFlags , std::vector< bool >* gtFlags ,
	std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher >& vertexTable,
	Real trimValue
)
{
	int sz = int( polygon.size() );
	std::vector< bool > gt( sz );
	int gtCount = 0;
	for( int j=0 ; j<sz ; j++ )
	{
		gt[j] = ( vertices[ polygon[j] ].template get<1>()>trimValue );
		if( gt[j] ) gtCount++;
	}
	if     ( gtCount==sz ){ if( gtPolygons ) gtPolygons->push_back( polygon ) ; if( gtFlags ) gtFlags->push_back( false ); }
	else if( gtCount==0  ){ if( ltPolygons ) ltPolygons->push_back( polygon ) ; if( ltFlags ) ltFlags->push_back( false ); }
	else
	{
		int start;
		for( start=0 ; start<sz ; start++ ) if( gt[start] && !gt[(start+sz-1)%sz] ) break;

		bool gtFlag = true;
		std::vector< Index > poly;

		// Add the initial vertex
		{
			int j1 = (start+int(sz)-1)%sz , j2 = start;
			Index v1 = polygon[j1] , v2 = polygon[j2] , vIdx;
			typename std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher >::iterator iter = vertexTable.find( EdgeKey< Index >(v1,v2) );
			if( iter==vertexTable.end() )
			{
				vertexTable[ EdgeKey< Index >(v1,v2) ] = vIdx = (Index)vertices.size();
				vertices.push_back( InterpolateVertices( vertices[v1] , vertices[v2] , trimValue ) );
			}
			else vIdx = iter->second;
			poly.push_back( vIdx );
		}

		for( int _j=0  ; _j<=sz ; _j++ )
		{
			int j1 = (_j+start+sz-1)%sz , j2 = (_j+start)%sz;
			Index v1 = polygon[j1] , v2 = polygon[j2];
			if( gt[j2]==gtFlag ) poly.push_back( v2 );
			else
			{
				Index vIdx;
				typename std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher >::iterator iter = vertexTable.find( EdgeKey< Index >(v1,v2) );
				if( iter==vertexTable.end() )
				{
					vertexTable[ EdgeKey< Index >(v1,v2) ] = vIdx = (Index)vertices.size();
					vertices.push_back( InterpolateVertices( vertices[v1] , vertices[v2] , trimValue ) );
				}
				else vIdx = iter->second;
				poly.push_back( vIdx );
				if( gtFlag ){ if( gtPolygons ) gtPolygons->push_back( poly ) ; if( ltFlags ) ltFlags->push_back( true ); }
				else        { if( ltPolygons ) ltPolygons->push_back( poly ) ; if( gtFlags ) gtFlags->push_back( true ); }
				poly.clear() , poly.push_back( vIdx ) , poly.push_back( v2 );
				gtFlag = !gtFlag;
			}
		}
	}
}

template< class Real , unsigned int Dim , typename Index , class Vertex >
void Triangulate( const std::vector< Vertex >& vertices , const std::vector< std::vector< Index > >& polygons , std::vector< std::vector< Index > >& triangles )
{
	triangles.clear();
	for( size_t i=0 ; i<polygons.size() ; i++ )
		if( polygons[i].size()>3 )
		{
			std::vector< Point< Real , Dim > > _vertices( polygons[i].size() );
			for( int j=0 ; j<int( polygons[i].size() ) ; j++ ) _vertices[j] = vertices[ polygons[i][j] ].template get<0>();
			std::vector< TriangleIndex< Index > > _triangles = MinimalAreaTriangulation< Index , Real , Dim >( ( ConstPointer( Point< Real , Dim > ) )GetPointer( _vertices ) , _vertices.size() );

			// Add the triangles to the mesh
			size_t idx = triangles.size();
			triangles.resize( idx+_triangles.size() );
			for( int j=0 ; j<int(_triangles.size()) ; j++ )
			{
				triangles[idx+j].resize(3);
				for( int k=0 ; k<3 ; k++ ) triangles[idx+j][k] = polygons[i][ _triangles[j].idx[k] ];
			}
		}
		else if( polygons[i].size()==3 ) triangles.push_back( polygons[i] );
}

template< class Real , unsigned int Dim , typename Index , class Vertex >
double PolygonArea( const std::vector< Vertex >& vertices , const std::vector< Index >& polygon )
{
	auto Area =[]( Point< Real , Dim > v1 , Point< Real , Dim > v2 , Point< Real , Dim > v3 )
	{
		Point< Real , Dim > v[] = { v2-v1 , v3-v1 };
		XForm< Real , 2 > Mass;
		for( int i=0 ; i<2 ; i++ ) for( int j=0 ; j<2 ; j++ ) Mass(i,j) = Point< Real , Dim >::Dot( v[i] , v[j] );
		double det = Mass.determinant();
		if( det<0 ) return (Real)0;
		else return (Real)( sqrt( Mass.determinant() ) / 2. );
	};

	if( polygon.size()<3 ) return 0.;
	else if( polygon.size()==3 ) return Area( vertices[polygon[0]].template get<0>() , vertices[polygon[1]].template get<0>() , vertices[polygon[2]].template get<0>() );
	else
	{
		Point< Real , DEFAULT_DIMENSION > center;
		for( size_t i=0 ; i<polygon.size() ; i++ ) center += vertices[ polygon[i] ].template get<0>();
		center /= Real( polygon.size() );
		double area = 0;
		for( size_t i=0 ; i<polygon.size() ; i++ ) area += Area( center , vertices[ polygon[i] ].template get<0>() , vertices[ polygon[ (i+1)%polygon.size() ] ].template get<0>() );
		return area;
	}
}

template< typename Index , class Vertex >
void RemoveHangingVertices( std::vector< Vertex >& vertices , std::vector< std::vector< Index > >& polygons )
{
	// Dense old->new vertex index map (keys are 0..vertices.size()-1) — a plain
	// vector replaces a per-vertex unordered_map lookup over millions of vertices.
	std::vector< Index > vMap( vertices.size() );
	std::vector< bool > vertexFlags( vertices.size() , false );
	for( size_t i=0 ; i<polygons.size() ; i++ ) for( size_t j=0 ; j<polygons[i].size() ; j++ ) vertexFlags[ polygons[i][j] ] = true;
	Index vCount = 0;
	for( Index i=0 ; i<(Index)vertices.size() ; i++ ) if( vertexFlags[i] ) vMap[i] = vCount++;
	for( size_t i=0 ; i<polygons.size() ; i++ ) for( size_t j=0 ; j<polygons[i].size() ; j++ ) polygons[i][j] = vMap[ polygons[i][j] ];

	std::vector< Vertex > _vertices( vCount );
	for( Index i=0 ; i<(Index)vertices.size() ; i++ ) if( vertexFlags[i] ) _vertices[ vMap[i] ] = vertices[i];
	vertices = _vertices;
}

template< typename Index >
void SetConnectedComponents( const std::vector< std::vector< Index > >& polygons , std::vector< std::vector< Index > >& components )
{
	// Connected components of the shared-(undirected-)edge graph over the polygons.
	// Same partition as the previous hash-table union-find (hence the same trim
	// result), but shared edges are found by sorting an (edge,polygon) array and
	// scanning equal runs — sequential memory instead of ~2 random hash probes per
	// edge. Union-find uses path-halving with the smaller index as the root.
	const size_t nPolys = polygons.size();
	std::vector< Index > parent( nPolys );
	for( size_t i=0 ; i<nPolys ; i++ ) parent[i] = (Index)i;

	auto Find = [&]( Index x )
	{
		while( parent[x]!=x ) x = parent[x] = parent[ parent[x] ];
		return x;
	};
	auto Union = [&]( Index a , Index b )
	{
		a = Find( a ) , b = Find( b );
		if( a!=b ){ if( a<b ) parent[b] = a ; else parent[a] = b; }
	};

	// Gather every undirected edge tagged with its owning polygon, then sort so the
	// (>=2) instances of a shared edge become adjacent.
	struct EdgeRec
	{
		Index lo , hi , poly;
		bool operator < ( const EdgeRec &e ) const { return lo<e.lo || ( lo==e.lo && hi<e.hi ); }
	};
	size_t edgeCount = 0;
	for( size_t i=0 ; i<nPolys ; i++ ) edgeCount += polygons[i].size();
	std::vector< EdgeRec > edges;
	edges.reserve( edgeCount );
	for( size_t i=0 ; i<nPolys ; i++ )
	{
		int sz = (int)polygons[i].size();
		for( int j=0 ; j<sz ; j++ )
		{
			Index a = polygons[i][j] , b = polygons[i][(j+1)%sz];
			EdgeRec e;
			e.lo = a<b ? a : b , e.hi = a<b ? b : a , e.poly = (Index)i;
			edges.push_back( e );
		}
	}
	std::sort( edges.begin() , edges.end() );
	for( size_t s=0 ; s<edges.size() ; )
	{
		size_t e = s+1;
		while( e<edges.size() && edges[e].lo==edges[s].lo && edges[e].hi==edges[s].hi ) e++;
		for( size_t k=s+1 ; k<e ; k++ ) Union( edges[s].poly , edges[k].poly );
		s = e;
	}

	// Flatten to roots, number components by their (smallest-index) representative,
	// and group polygons in ascending order. A dense root->component vector avoids
	// the per-polygon hash lookups the previous unordered_map vMap incurred.
	std::vector< Index > roots( nPolys );
	for( size_t i=0 ; i<nPolys ; i++ ) roots[i] = Find( (Index)i );
	std::vector< Index > compOf( nPolys );
	int cCount = 0;
	for( size_t i=0 ; i<nPolys ; i++ ) if( roots[i]==(Index)i ) compOf[i] = (Index)( cCount++ );
	components.resize( cCount );
	for( size_t i=0 ; i<nPolys ; i++ ) components[ compOf[ roots[i] ] ].push_back( (Index)i );
}

// Core trimming pipeline, factored out of Execute() so it can be driven either by
// the command-line tool (read/write PLY) or in-process from OpenMVS (PoissonReconLib).
// Takes the vertices (position + per-vertex density value) and polygons in memory,
// modifies `vertices` in place (split vertices added, then hanging vertices removed)
// and returns the kept polygons in `outPolygons`. Identical logic to the original
// Execute body; the only change is parameters instead of CmdLine globals / PLY I/O.
template< typename Real , unsigned int Dim , typename Index , typename Vertex >
void TrimMeshInMemory
(
	std::vector< Vertex > &vertices ,
	const std::vector< std::vector< Index > > &polygons ,
	Real trimValue , Real islandAreaRatio , bool removeIslands , bool polygonMesh , bool debug ,
	std::vector< std::vector< Index > > &outPolygons
)
{
	std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher > vertexTable;
	std::vector< std::vector< Index > > ltPolygons , gtPolygons;
	std::vector< bool > ltFlags , gtFlags;

#if PR_TRIM_PROFILE
	double _tp = Time();
	std::cout << "[TRIM-PROFILE] input: verts=" << vertices.size() << " polys=" << polygons.size() << std::endl;
#endif // PR_TRIM_PROFILE
	vertexTable.reserve( polygons.size() );
	for( size_t i=0 ; i<polygons.size() ; i++ ) SplitPolygon( polygons[i] , vertices , &ltPolygons , &gtPolygons , &ltFlags , &gtFlags , vertexTable , trimValue );
#if PR_TRIM_PROFILE
	std::cout << "[TRIM-PROFILE] SplitPolygon: " << Time()-_tp << " (s)" << std::endl; _tp = Time();
#endif // PR_TRIM_PROFILE

	if( islandAreaRatio>0 )
	{
		std::vector< std::vector< Index > > _polygons , _components;
		size_t gtComponentStart;
		{
			std::vector< std::vector< Index > > ltComponents , gtComponents;
			SetConnectedComponents( ltPolygons , ltComponents );
			SetConnectedComponents( gtPolygons , gtComponents );
#if PR_TRIM_PROFILE
			std::cout << "[TRIM-PROFILE]   .. SetConnComp x2: " << Time()-_tp << " (s)" << std::endl; _tp = Time();
#endif // PR_TRIM_PROFILE
			gtComponentStart = ltComponents.size();
			for( unsigned int i=0 ; i<gtComponents.size() ; i++ ) for( unsigned int j=0 ; j<gtComponents[i].size() ; j++ ) gtComponents[i][j] += (Index)ltPolygons.size();

			_polygons.reserve( ltPolygons.size() + gtPolygons.size() );
			_components.reserve( ltComponents.size() + gtComponents.size() );
			_polygons.insert( _polygons.end() , std::make_move_iterator( ltPolygons.begin() ) , std::make_move_iterator( ltPolygons.end() ) );
			_polygons.insert( _polygons.end() , std::make_move_iterator( gtPolygons.begin() ) , std::make_move_iterator( gtPolygons.end() ) );
			_components.insert( _components.end() , std::make_move_iterator( ltComponents.begin() ) , std::make_move_iterator( ltComponents.end() ) );
			_components.insert( _components.end() , std::make_move_iterator( gtComponents.begin() ) , std::make_move_iterator( gtComponents.end() ) );
		}
		std::vector< typename ComponentGraph< Index >::Node > nodes( _components.size() );

		for( unsigned int i=0 ; i<_components.size() ; i++ )
		{
			for( size_t j=0 ; j<_components[i].size() ; j++ ) nodes[i].area += PolygonArea< Real , Dim , Index , Vertex >( vertices , _polygons[ _components[i][j] ] );
			nodes[i].polygonIndices.push_back( _components[i] );
		}
#if PR_TRIM_PROFILE
		std::cout << "[TRIM-PROFILE]   .. Copy+Area: " << Time()-_tp << " (s)" << std::endl; _tp = Time();
#endif // PR_TRIM_PROFILE

		struct _HalfEdge
		{
			Index lo , hi , comp;
			bool operator < ( const _HalfEdge &e ) const { return lo<e.lo || ( lo==e.lo && hi<e.hi ); }
		};
		std::vector< _HalfEdge > halfEdges;
		{
			size_t _heTotal = 0;
			for( unsigned int i=0 ; i<_components.size() ; i++ ) for( unsigned int j=0 ; j<_components[i].size() ; j++ ) _heTotal += _polygons[ _components[i][j] ].size();
			halfEdges.reserve( _heTotal );
		}
		for( unsigned int i=0 ; i<_components.size() ; i++ ) for( unsigned int j=0 ; j<_components[i].size() ; j++ )
		{
			const std::vector< Index > &poly = _polygons[ _components[i][j] ];
			int sz = (int)poly.size();
			for( int k=0 ; k<sz ; k++ )
			{
				Index a = poly[k] , b = poly[ (k+1)%sz ];
				_HalfEdge e;
				e.lo = a<b ? a : b , e.hi = a<b ? b : a , e.comp = (Index)i;
				halfEdges.push_back( e );
			}
		}
		std::sort( halfEdges.begin() , halfEdges.end() );
#if PR_TRIM_PROFILE
		std::cout << "[TRIM-PROFILE]   .. BoundaryHalfEdges: " << Time()-_tp << " (s)" << std::endl; _tp = Time();
#endif // PR_TRIM_PROFILE

		std::unordered_set< EdgeKey< Index > , typename EdgeKey< Index >::Hasher > componentEdges;
		for( size_t s=0 ; s<halfEdges.size() ; )
		{
			size_t e = s+1;
			while( e<halfEdges.size() && halfEdges[e].lo==halfEdges[s].lo && halfEdges[e].hi==halfEdges[s].hi ) e++;
			for( size_t a=s ; a<e ; a++ ) for( size_t b=a+1 ; b<e ; b++ ) if( halfEdges[a].comp!=halfEdges[b].comp )
			{
				componentEdges.insert( EdgeKey< Index >( halfEdges[a].comp , halfEdges[b].comp ) );
				componentEdges.insert( EdgeKey< Index >( halfEdges[b].comp , halfEdges[a].comp ) );
			}
			s = e;
		}
		for( auto iter=componentEdges.begin() ; iter!=componentEdges.end() ; iter++ )
		{
			nodes[ iter->key1 ].neighbors.push_back( &nodes[ iter->key2 ] );
			nodes[ iter->key2 ].neighbors.push_back( &nodes[ iter->key1 ] );
		}
		if( debug ) ComponentGraph< Index >::SanityCheck( nodes.size() , [&]( size_t i ){ return &nodes[i]; } );

		double area = 0;
		for( unsigned int i=0 ; i<nodes.size() ; i++ ) area += nodes[i].area;
#if PR_TRIM_PROFILE
		std::cout << "[TRIM-PROFILE] ComponentEdges+area: " << Time()-_tp << " (s) , components=" << nodes.size() << std::endl; _tp = Time();
		size_t _nMerges = 0;
#endif // PR_TRIM_PROFILE

		bool done = false;
		while( !done )
		{
			done = true;
			unsigned int idx = -1;
			for( unsigned int i=0 ; i<nodes.size() ; i++ ) if( nodes[i].polygonIndices.size() && nodes[i].neighbors.size() ) if( idx==-1 || nodes[i].area<nodes[idx].area ) idx = i;
			if( idx!=-1 && nodes[idx].area<area*islandAreaRatio )
			{
				nodes[idx].merge();
				done = false;
#if PR_TRIM_PROFILE
				_nMerges++;
#endif // PR_TRIM_PROFILE
				if( debug ) ComponentGraph< Index >::SanityCheck( nodes.size() , [&]( size_t i ){ return &nodes[i]; } );
			}
		}
#if PR_TRIM_PROFILE
		std::cout << "[TRIM-PROFILE] IslandMerge loop: " << Time()-_tp << " (s) , merges=" << _nMerges << std::endl; _tp = Time();
#endif // PR_TRIM_PROFILE

		ltPolygons.clear() , gtPolygons.clear();

		for( unsigned int i=0 ; i<gtComponentStart ; i++ )
			if( !nodes[i].neighbors.size() && nodes[i].area<area*islandAreaRatio && removeIslands ) ; // small island
			else for( auto iter=nodes[i].polygonIndices.begin() ; iter!=nodes[i].polygonIndices.end() ; iter++ ) for( size_t j=0 ; j<iter->size() ; j++ ) ltPolygons.push_back( _polygons[ (*iter)[j] ] );
		for( unsigned int i=(unsigned int)gtComponentStart ; i<nodes.size() ; i++ )
			if( !nodes[i].neighbors.size() && nodes[i].area<area*islandAreaRatio && removeIslands ) ; // small island
			else for( auto iter=nodes[i].polygonIndices.begin() ; iter!=nodes[i].polygonIndices.end() ; iter++ ) for( size_t j=0 ; j<iter->size() ; j++ ) gtPolygons.push_back( _polygons[ (*iter)[j] ] );
	}

	if( !polygonMesh )
	{
		{
			std::vector< std::vector< Index > > polys = ltPolygons;
			Triangulate< Real , Dim , Index , Vertex >( vertices , ltPolygons , polys ) , ltPolygons = polys;
		}
		{
			std::vector< std::vector< Index > > polys = gtPolygons;
			Triangulate< Real , Dim , Index , Vertex >( vertices , gtPolygons , polys ) , gtPolygons = polys;
		}
	}

	RemoveHangingVertices( vertices , gtPolygons );
#if PR_TRIM_PROFILE
	std::cout << "[TRIM-PROFILE] Triangulate+RemoveHanging: " << Time()-_tp << " (s)" << std::endl;
#endif // PR_TRIM_PROFILE
	outPolygons = std::move( gtPolygons );
}

namespace PoissonReconLib
{
	bool Trim( const Mesh &in , const TrimParams &params , Mesh &out )
	{
		typedef float Real;
		static const unsigned int Dim = 3;
		typedef int Index;
		typedef VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::ValueFactory< Real > > Factory;
		typedef typename Factory::VertexType Vertex;

		out.Clear();
		const size_t nv = in.VertexCount() , nt = in.TriangleCount();
		if( nv==0 || nt==0 ) return false;

		std::vector< Vertex > vertices( nv );
		for( size_t i=0 ; i<nv ; i++ )
		{
			const float *v = &in.vertices[ i*4 ];
			vertices[i].template get<0>()[0] = (Real)v[0];
			vertices[i].template get<0>()[1] = (Real)v[1];
			vertices[i].template get<0>()[2] = (Real)v[2];
			vertices[i].template get<1>()    = (Real)v[3]; // density -> value
		}
		std::vector< std::vector< Index > > polygons( nt );
		for( size_t i=0 ; i<nt ; i++ )
		{
			const uint32_t *tr = &in.triangles[ i*3 ];
			polygons[i] = { (Index)tr[0] , (Index)tr[1] , (Index)tr[2] };
		}

		Real effectiveTrim = (Real)params.trim;
#if PR_DENSITY_BAND_FILTER
		// Density-band filter (see macro block near top). Drop the low-confidence
		// down-facing "underside/drape" faces whose density lies in the band
		// [trim, trim+width) BEFORE the normal trim runs. A face is "low-confidence"
		// only if even its STRONGEST vertex density (dMax) is still below trim+width
		// (so genuine transition faces climbing into solid density are kept). Among
		// those, drop only faces whose unit normal.z < minUp. Confident faces and
		// sub-trim faces (removed by the trim itself) are untouched.
		{
			const Real bandLo = (Real)( PR_DENSITY_BAND_LO_X100 / 100.0 );
			const Real bandHi = (Real)( PR_DENSITY_BAND_HI_X100 / 100.0 );
			const Real minUp  = (Real)( PR_DENSITY_BAND_MIN_UP_X100 / 100.0 );
			// The band filter needs the trimmer to KEEP the band [LO,HI); if the
			// caller passed a --trim above LO it would delete the whole band (both
			// the underside AND the genuine up-facing fringe), throwing away this
			// filter's work. Clamp the effective trim down to LO so completeness is
			// preserved and the filter (not the trim) removes the buried underside.
			// No effect if the caller already passed --trim <= LO.
			if( effectiveTrim > bandLo ) {
				std::cout << "[DENSITY-BAND] effective trim lowered " << effectiveTrim << " -> " << bandLo << " (filter keeps the band; underside removed by occlusion gate)" << std::endl;
				effectiveTrim = bandLo;
			}
			const size_t NVv = vertices.size();
			// Pass 1: accumulate AREA-WEIGHTED face normals into per-vertex normals,
			// so the orientation test uses a neighborhood-SMOOTHED normal (robust to
			// noisy/flipped single triangles at the rim -- those got flat faces cut).
			std::vector< Real > vnX( NVv, (Real)0 ), vnY( NVv, (Real)0 ), vnZ( NVv, (Real)0 );
			for( size_t i=0 ; i<polygons.size() ; i++ )
			{
				const std::vector< Index > &poly = polygons[i];
				if( poly.size()!=3 ) continue;
				const auto &A = vertices[ poly[0] ].template get<0>();
				const auto &B = vertices[ poly[1] ].template get<0>();
				const auto &C = vertices[ poly[2] ].template get<0>();
				const Real ux=B[0]-A[0] , uy=B[1]-A[1] , uz=B[2]-A[2];
				const Real vx=C[0]-A[0] , vy=C[1]-A[1] , vz=C[2]-A[2];
				const Real nx=uy*vz-uz*vy , ny=uz*vx-ux*vz , nz=ux*vy-uy*vx; // ~2*area weighted
				for( int k=0 ; k<3 ; k++ ){ vnX[poly[k]]+=nx; vnY[poly[k]]+=ny; vnZ[poly[k]]+=nz; }
			}
#if PR_DENSITY_BAND_OCCLUSION
			// Build a top-down max-Z reference surface from CONFIDENT vertices ONLY
			// (density >= bandHi = the OBSERVED surface). A band face hanging far below
			// this is an unobserved underside outlier. Using confident verts (not all)
			// stops a lobe's own low-density verts from defining its local top.
			Real gMinX=(Real)1e30,gMinY=(Real)1e30,gMaxX=(Real)-1e30,gMaxY=(Real)-1e30;
			for( size_t v=0 ; v<NVv ; v++ ){ const auto &P=vertices[v].template get<0>(); if(P[0]<gMinX)gMinX=P[0]; if(P[1]<gMinY)gMinY=P[1]; if(P[0]>gMaxX)gMaxX=P[0]; if(P[1]>gMaxY)gMaxY=P[1]; }
			Real medianEdge=(Real)0;
			{
				std::vector< Real > el; el.reserve( 100000 );
				for( size_t i=0 ; i<polygons.size() && el.size()<100000 ; i++ ){ const std::vector< Index > &p=polygons[i]; if(p.size()!=3) continue; const auto &A=vertices[p[0]].template get<0>(); const auto &B=vertices[p[1]].template get<0>(); const Real dx=B[0]-A[0],dy=B[1]-A[1],dz=B[2]-A[2]; el.push_back( (Real)sqrt((double)(dx*dx+dy*dy+dz*dz)) ); }
				if(!el.empty()){ std::nth_element(el.begin(),el.begin()+el.size()/2,el.end()); medianEdge=el[el.size()/2]; }
			}
			const Real spanX=std::max((Real)1e-6,gMaxX-gMinX), spanY=std::max((Real)1e-6,gMaxY-gMinY);
			Real cell=(Real)(PR_DENSITY_BAND_CELL_MULT_X100/100.0)*medianEdge;
			if( cell<=(Real)0 ) cell=std::max(spanX,spanY);
			const int MAXDIM=4096;
			int gw=(int)(spanX/cell)+2, gh=(int)(spanY/cell)+2;
			if( gw>MAXDIM || gh>MAXDIM ){ cell=std::max(spanX/(MAXDIM-2),spanY/(MAXDIM-2)); gw=(int)(spanX/cell)+2; gh=(int)(spanY/cell)+2; }
			const Real invCell=(Real)1/cell;
			const Real buryMargin=(Real)(PR_DENSITY_BAND_BURY_MULT_X100/100.0)*medianEdge;
			std::vector< Real > topZ( (size_t)gw*(size_t)gh , (Real)-1e30 );
			auto cellIdx=[&]( Real x , Real y )->size_t{ int cx=(int)((x-gMinX)*invCell); if(cx<0)cx=0; if(cx>=gw)cx=gw-1; int cy=(int)((y-gMinY)*invCell); if(cy<0)cy=0; if(cy>=gh)cy=gh-1; return (size_t)cy*(size_t)gw+(size_t)cx; };
			size_t nConf=0;
			for( size_t v=0 ; v<NVv ; v++ ){ if( vertices[v].template get<1>() < bandHi ) continue; const auto &P=vertices[v].template get<0>(); Real &t=topZ[cellIdx(P[0],P[1])]; if(P[2]>t) t=P[2]; ++nConf; }
#endif // PR_DENSITY_BAND_OCCLUSION
			std::vector< std::vector< Index > > kept;
			kept.reserve( polygons.size() );
			// diagnostics
			size_t dropped = 0, bandFaces = 0, bandDown = 0, bandVert = 0, bandUp = 0;
			size_t candBuried = 0, candSurface = 0; // down-facing band candidates split by occlusion
			Real dAllMin = (Real)1e30, dAllMax = (Real)-1e30;
			// histogram of NON-up-facing (down+vertical) faces per 1.0-density bucket
			// (ABSOLUTE density), to reveal where the errant underside/drape lives.
			static const int NB = 16;
			size_t histAll[NB] = {0}, histNonUp[NB] = {0};
			for( size_t i=0 ; i<polygons.size() ; i++ )
			{
				const std::vector< Index > &poly = polygons[i];
				bool drop = false;
				if( poly.size()==3 )
				{
					const Real d0 = vertices[ poly[0] ].template get<1>();
					const Real d1 = vertices[ poly[1] ].template get<1>();
					const Real d2 = vertices[ poly[2] ].template get<1>();
					const Real dMax = std::max( d0 , std::max( d1 , d2 ) );
					const Real dMinF = std::min( d0 , std::min( d1 , d2 ) );
					if( dMinF<dAllMin ) dAllMin = dMinF;
					if( dMax >dAllMax ) dAllMax = dMax;
					// SMOOTHED orientation: sum of the 3 per-vertex (area-weighted)
					// normals -> robust up-ness. A flat edge whose vertices touch many
					// up-facing faces stays up even if one incident triangle is bad.
					const Real sx = vnX[poly[0]]+vnX[poly[1]]+vnX[poly[2]];
					const Real sy = vnY[poly[0]]+vnY[poly[1]]+vnY[poly[2]];
					const Real sz = vnZ[poly[0]]+vnZ[poly[1]]+vnZ[poly[2]];
					const Real sl = (Real)sqrt( (double)( sx*sx+sy*sy+sz*sz ) );
					const Real unz = sl>(Real)1e-12 ? (sz/sl) : (Real)0;
					// absolute-density-bucket histogram
					int b = (int)dMax; if( b<0 ) b=0; if( b>=NB ) b=NB-1;
					histAll[b]++;
					if( unz < (Real)0.10 ) histNonUp[b]++;
					// ABSOLUTE band membership: face's BEST vertex density in [LO,HI)
					if( dMax>=bandLo && dMax<bandHi )
					{
						++bandFaces;
						if     ( unz < (Real)-0.10 ) ++bandDown;
						else if ( unz < (Real) 0.10 ) ++bandVert;
						else                          ++bandUp;
#if PR_DENSITY_BAND_OCCLUSION
						// HEIGHT-OUTLIER test (the "flip + remove outliers" idea): drop a
						// band face whose centroid hangs more than buryMargin BELOW the
						// LOCAL GROUND at its XY. Local ground = LOWEST confident-surface top
						// over a small cell neighborhood (not the same-cell canopy max), so
						// ground under canopy (at local ground) is spared while lobes (below
						// local ground) are cut. No orientation requirement.
						const Real cxx=(vertices[poly[0]].template get<0>()[0]+vertices[poly[1]].template get<0>()[0]+vertices[poly[2]].template get<0>()[0])/(Real)3;
						const Real cyy=(vertices[poly[0]].template get<0>()[1]+vertices[poly[1]].template get<0>()[1]+vertices[poly[2]].template get<0>()[1])/(Real)3;
						const Real czz=(vertices[poly[0]].template get<0>()[2]+vertices[poly[1]].template get<0>()[2]+vertices[poly[2]].template get<0>()[2])/(Real)3;
						int ccx=(int)((cxx-gMinX)*invCell); if(ccx<0)ccx=0; if(ccx>=gw)ccx=gw-1;
						int ccy=(int)((cyy-gMinY)*invCell); if(ccy<0)ccy=0; if(ccy>=gh)ccy=gh-1;
						Real localGround=(Real)1e30;
						for( int dy=-(PR_DENSITY_BAND_NBR_RADIUS) ; dy<=(PR_DENSITY_BAND_NBR_RADIUS) ; ++dy )
							for( int dx=-(PR_DENSITY_BAND_NBR_RADIUS) ; dx<=(PR_DENSITY_BAND_NBR_RADIUS) ; ++dx ) {
								const int nx=ccx+dx, ny=ccy+dy;
								if( nx<0 || ny<0 || nx>=gw || ny>=gh ) continue;
								const Real t=topZ[(size_t)ny*(size_t)gw+(size_t)nx];
								if( t>(Real)-1e29 && t<localGround ) localGround=t;
							}
						if( localGround<(Real)1e29 && (localGround-czz)>buryMargin ) { drop = true; ++candBuried; }
						else ++candSurface;
#endif
					}
				}
				if( drop ) dropped++;
				else kept.push_back( poly );
			}
			std::cout << "[DENSITY-BAND] tris=" << polygons.size()
				<< " density=[" << dAllMin << "," << dAllMax << "]"
				<< " trim=" << (Real)params.trim << " band=[" << bandLo << "," << bandHi << ")"
				<< " (height-outlier vs confident surface, buryMargin=" << buryMargin << ")"
				<< " | band-faces=" << bandFaces
				<< " (down=" << bandDown << " vert=" << bandVert << " up=" << bandUp << ")"
				<< " | below-observed=" << candBuried << " on-surface-spared=" << candSurface
				<< " -> dropped=" << dropped << std::endl;
			std::cout << "[DENSITY-BAND] non-up faces by ABSOLUTE density bucket:" << std::endl;
			for( int b=0 ; b<NB ; b++ ) if( histAll[b] )
				std::cout << "    d[" << b << "," << (b+1) << (b==NB-1?"+":"") << ") all=" << histAll[b] << " non-up=" << histNonUp[b] << std::endl;
			if( dropped>0 )
				polygons = std::move( kept );
		}
#endif // PR_DENSITY_BAND_FILTER

		std::vector< std::vector< Index > > outPolygons;
		try
		{
			TrimMeshInMemory< Real , Dim , Index , Vertex >( vertices , polygons , effectiveTrim , (Real)params.aRatio , params.removeIslands , false /*polygonMesh*/ , false /*debug*/ , outPolygons );
		}
		catch( const std::exception &e )
		{
			MK_WARN( "PoissonReconLib::Trim failed: " , e.what() );
			out.Clear();
			return false;
		}

		out.vertices.resize( vertices.size()*4 );
		for( size_t i=0 ; i<vertices.size() ; i++ )
		{
			out.vertices[ i*4+0 ] = (float)vertices[i].template get<0>()[0];
			out.vertices[ i*4+1 ] = (float)vertices[i].template get<0>()[1];
			out.vertices[ i*4+2 ] = (float)vertices[i].template get<0>()[2];
			out.vertices[ i*4+3 ] = (float)vertices[i].template get<1>();
		}
		out.triangles.reserve( outPolygons.size()*3 );
		for( size_t i=0 ; i<outPolygons.size() ; i++ )
		{
			const std::vector< Index > &p = outPolygons[i];
			if( p.size()==3 ) out.triangles.push_back( (uint32_t)p[0] ) , out.triangles.push_back( (uint32_t)p[1] ) , out.triangles.push_back( (uint32_t)p[2] );
			else for( size_t k=2 ; k<p.size() ; k++ ) out.triangles.push_back( (uint32_t)p[0] ) , out.triangles.push_back( (uint32_t)p[k-1] ) , out.triangles.push_back( (uint32_t)p[k] );
		}
		return out.TriangleCount()>0;
	}
}

#ifndef POISSONRECON_LIB_BUILD
template< typename Real , unsigned int Dim , typename Index , typename ... AuxDataFactories >
int Execute( AuxDataFactories ... auxDataFactories )
{
	typedef VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , Dim > , typename VertexFactory::ValueFactory< Real > , AuxDataFactories ... > Factory;
	typedef typename Factory::VertexType Vertex;
	typename VertexFactory::PositionFactory< Real , Dim > pFactory;
	typename VertexFactory::ValueFactory< Real > vFactory;
	Factory factory( pFactory , vFactory , auxDataFactories ... );
	Real min , max;

	std::vector< Vertex > vertices;
	std::vector< std::vector< Index > > polygons;

	int ft;
	std::vector< std::string > comments;
	PLY::ReadPolygons< Factory , Index >( In.value , factory , vertices , polygons , ft , comments );

	min = max = vertices[0].template get<1>();
	for( size_t i=0 ; i<vertices.size() ; i++ ) min = std::min< Real >( min , vertices[i].template get<1>() ) , max = std::max< Real >( max , vertices[i].template get<1>() );

	if( Verbose.set )
	{
		std::cout << "*********************************************" << std::endl;
		std::cout << "*********************************************" << std::endl;
		std::cout << "** Running Surface Trimmer (Version " << ADAPTIVE_SOLVERS_VERSION << ") **" << std::endl;
		std::cout << "*********************************************" << std::endl;
		std::cout << "*********************************************" << std::endl;
	}
	char str[1024];
	for( int i=0 ; params[i] ; i++ )
		if( params[i]->set )
		{
			params[i]->writeValue( str );
			if( Verbose.set )
			{
				if( strlen( str ) ) std::cout << "\t--" << params[i]->name << " " << str << std::endl;
				else                std::cout << "\t--" << params[i]->name << std::endl;
			}
		}
	if( Verbose.set ) printf( "Value Range: [%f,%f]\n" , min , max );

	std::vector< std::vector< Index > > outPolygons;
	TrimMeshInMemory< Real , Dim , Index , Vertex >( vertices , polygons , (Real)Trim.value , (Real)IslandAreaRatio.value , RemoveIslands.set , PolygonMesh.set , Debug.set , outPolygons );

	if( Out.set ) PLY::WritePolygons( Out.value , factory , vertices , outPolygons , ASCII.set ? PLY_ASCII : ft , comments );

	return EXIT_SUCCESS;
}
int main( int argc , char* argv[] )
{
	setvbuf( stdout , NULL , _IONBF , 0 ); // unbuffered stdout so progress streams live when output is piped/captured
	CmdLineParse( argc-1 , &argv[1] , params );

	if( !In.set || !Trim.set )
	{
		ShowUsage( argv[0] );
		return EXIT_FAILURE;
	}
	typedef float Real;
	static constexpr unsigned int Dim = DEFAULT_DIMENSION;
	typedef VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , Dim > , typename VertexFactory::ValueFactory< Real > > Factory;
	Factory factory;
	bool *readFlags = new bool[ factory.plyReadNum() ];
	std::vector< PlyProperty > unprocessedProperties;
	size_t vNum;
	PLY::ReadVertexHeader( In.value , factory , readFlags , unprocessedProperties , vNum );
	if( vNum>std::numeric_limits< int >::max() )
	{
		if( !Long.set ) MK_WARN( "Number of vertices not supported by 32-bit indexing. Switching to 64-bit indexing" );
		Long.set = true;
	}
	if( !factory.template plyValidReadProperties<0>( readFlags ) ) MK_THROW( "Ply file does not contain positions" );
	if( !factory.template plyValidReadProperties<1>( readFlags ) ) MK_THROW( "Ply file does not contain values" );
	delete[] readFlags;

	if( Long.set ) return Execute< Real , Dim , long long >( VertexFactory::DynamicFactory< Real >( unprocessedProperties ) );
	else           return Execute< Real , Dim , int       >( VertexFactory::DynamicFactory< Real >( unprocessedProperties ) );

}
#endif // POISSONRECON_LIB_BUILD
