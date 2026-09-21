/*
Copyright (c) 2017, Michael Kazhdan
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
#ifndef MULTI_THREADING_INCLUDED
#define MULTI_THREADING_INCLUDED

#include <thread>
#include <vector>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <future>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

namespace PoissonRecon
{
	struct ThreadPool
	{
		enum ParallelType
		{
#ifdef _OPENMP
			OPEN_MP ,
#endif // _OPENMP
			ASYNC ,
			NONE
		};
		static const std::vector< std::string > ParallelNames;

		enum ScheduleType
		{
			STATIC ,
			DYNAMIC
		};
		static const std::vector< std::string > ScheduleNames;

		static unsigned int NumThreads( void ){ return _NumThreads; }

		// Count of ParallelSections invocations. ParallelSections does NOT use the pool --
		// it spawns via std::async(std::launch::async) and heap-allocates a futures vector
		// per call (see _ParallelSections) -- and the level-set extractor calls it inside a
		// per-depth loop for every slab, so the count is expected to be O(1e5) per run.
		// Relaxed atomic: ~1e5 increments against thread spawns is free, and callers are
		// not all on one thread.
		static std::atomic< size_t > SectionCount;
		// Set the worker count for every subsequent ParallelFor/ParallelSections.
		// MUST be called BEFORE a solve starts: callers size per-thread scratch from
		// NumThreads() exactly once (FEMTree's densityKeys/dataKeys, PoissonReconLib's
		// per-thread vertex/face buffers), so raising it mid-solve would leave those
		// arrays short and index out of bounds. n==0 leaves the hardware_concurrency
		// default untouched. There is no Init() in this fork -- this is the only way
		// to override the default.
		static void SetNumThreads( unsigned int n ){ if( n ) _NumThreads = n; }
		static ParallelType ParallelizationType;
		static size_t ChunkSize;
		static ScheduleType Schedule;
		// Ranges shorter than this run on the calling thread instead of opening a
		// parallel region. See the cutoff in ParallelFor for what it is for, why it is
		// a heuristic, and how to calibrate it. Set to 0 to disable (A/B switch).
		static size_t SerialCutoff;

		template< typename Function , typename ... Functions >
		static void ParallelSections( const Function &function , const Functions & ... functions )
		{
			std::vector< std::future< void > > futures;
			if constexpr( sizeof ... (Functions) )
			{
				futures.reserve( sizeof...(Functions) );
				_ParallelSections( futures , functions... );
			}
			function();
			for( unsigned int i=0 ; i<futures.size() ; i++ ) futures[i].get();
		}

		template< typename Function , typename ... Functions >
		static void ParallelSections( const Function &&function , const Functions && ... functions )
		{
			std::vector< std::future< void > > futures;
			if constexpr( sizeof ... (Functions) )
			{
				futures.reserve( sizeof...(Functions) );
				_ParallelSections( futures , std::move(functions)... );
			}
			function();
			for( unsigned int i=0 ; i<futures.size() ; i++ ) futures[i].get();
		}

		// MEASURED AND REVERTED -- do not re-template this on the body.
		//
		// The body arrives as a type-erased std::function and is invoked once per
		// element below, so templating ParallelFor on the closure type looks like an
		// obvious win: the call could then inline, vectorize, and keep state in
		// registers. It was tried (all 182 call sites pass inline lambdas, so none had
		// to change) and measured on MechanicFalls against the run directly before it:
		//
		//                      before      after templating
		//   normal field         3.4 s      3.5 s
		//   linear solve         2.3 s      2.2 s
		//   extraction           2.7 s      2.7 s   (sub-phases byte-identical)
		//   SOLVE TOTAL         11.6 s     11.6 s
		//   surface stage      14.49 s    14.70 s
		//
		// Nothing, inside a +/-0.2 s noise floor. Two reasons, both of which also say
		// not to try the same trick elsewhere in here:
		//   * this target builds with LTCG (/GL) and ParallelFor is header-defined with
		//     inline-lambda call sites, so the linker could already see the concrete
		//     types -- the indirection was substantially gone already;
		//   * the bodies are MEMORY-bound, not call-bound. Splatting samples into an 8M
		//     node octree or gathering neighbours in a Gauss-Seidel sweep is pointer
		//     chasing; against a cache miss, an indirect call is noise.
		//
		// The cost was real: 182 instantiations across two TUs, compile time, object
		// size, and a deeper fork from upstream. Reverted on that trade.
		static void ParallelFor( size_t begin , size_t end , const std::function< void ( unsigned int , size_t ) > &iterationFunction , unsigned int numThreads=_NumThreads , ParallelType pType=ParallelizationType , ScheduleType schedule=Schedule , size_t chunkSize=ChunkSize )
		{
			if( begin>=end ) return;
			size_t range = end - begin;
			size_t chunks = ( range + chunkSize - 1 ) / chunkSize;

			// If the computation is serial, go ahead and run it
			if( pType==ParallelType::NONE || numThreads<=1 )
			{
				for( size_t i=begin ; i<end ; i++ ) iterationFunction( 0 , i );
				return;
			}

			// If the chunkSize is too large to satisfy all the threads, lower it
			if( range<=chunkSize*(numThreads-1) )
			{
				chunkSize = ( range + numThreads - 1 ) / numThreads;
				chunks = numThreads = (unsigned int)( ( range + chunkSize - 1 ) / chunkSize );
			}

			std::function< void (unsigned int , size_t ) > _ChunkFunction = [ &iterationFunction , begin , end , chunkSize ]( unsigned int thread , size_t chunk )
				{
					const size_t _begin = begin + chunkSize*chunk;
					const size_t _end = std::min< size_t >( end , _begin+chunkSize );
					for( size_t i=_begin ; i<_end ; i++ ) iterationFunction( thread , i );
				};

			// SMALL-RANGE CUTOFF.
			//
			// Every call above a range of 1 used to open a parallel region. Fork/join plus
			// the closing barrier on a 32-thread region costs roughly 5-20 us, which a short
			// loop of cheap work cannot come close to repaying -- and the solver issues a
			// great many short loops, because the sliced Gauss-Seidel relaxes one slice at a
			// time and a coarse level has few nodes per slice.
			//
			// MEASURED on MechanicFalls (depth 10, 8.0M nodes), per-node solve cost by level:
			//   depth  5:  35,937 nodes  ->  9.4 us/node
			//   depth  6:  59,024 nodes  ->  8.9 us/node
			//   depth 10: 4,727,992 nodes -> 0.26 us/node
			// A 36x per-node gap that scales INVERSELY with problem size is the signature of
			// fixed per-call overhead, not of the arithmetic. Depths 5-7 together were 1.42 s
			// for 3.7% of the nodes.
			//
			// THE THREAD INDEX IS PRESERVED, deliberately. Bodies reduce into per-thread
			// slots -- `scratch[thread] += Dot(r[i],r[i])` and friends throughout
			// SparseMatrixInterface.inl -- and the caller then sums those slots. Running the
			// plain `iterationFunction(0,i)` loop would put every term in slot 0 and change
			// the summation order, hence the floating-point result. Walking the SAME chunks
			// and handing each the thread a STATIC schedule would have given it (chunk %
			// numThreads, exactly what `schedule(static,1)` does) keeps the grouping -- so
			// this is bit-identical to the parallel path under STATIC scheduling, and no
			// worse than the existing run-to-run variation under DYNAMIC.
			//
			// HEURISTIC, AND ONLY ON COUNT: this cannot tell 1000 cheap iterations from 1000
			// expensive ones, so a low cutoff could serialize a short loop whose body is
			// costly. Calibrate against the per-level "Updated constraints / Got system /
			// Solved in" lines -- the coarse depths should compress while depth 10 barely
			// moves. SerialCutoff = 0 disables it for a clean A/B.
			if( SerialCutoff && range<SerialCutoff )
			{
				for( size_t c=0 ; c<chunks ; c++ ) _ChunkFunction( (unsigned int)( c % numThreads ) , c );
				return;
			}

			if( false ){}
#ifdef _OPENMP
			else if( pType==ParallelType::OPEN_MP )
			{
				if( schedule==ScheduleType::STATIC )
#pragma omp parallel for num_threads( numThreads ) schedule( static , 1 )
					for( int c=0 ; c<chunks ; c++ ) _ChunkFunction( omp_get_thread_num() , c );
				else if( schedule==ScheduleType::DYNAMIC )
#pragma omp parallel for num_threads( numThreads ) schedule( dynamic , 1 )
					for( int c=0 ; c<chunks ; c++ ) _ChunkFunction( omp_get_thread_num() , c );
			}
#endif // _OPENMP
			else if( pType==ParallelType::ASYNC )
			{
				// These live HERE because this is the only branch that reads them. They used
				// to be built unconditionally, above the dispatch -- so on the OpenMP path,
				// which is the one this build takes (_OPENMP is defined, making
				// ParallelType(0) == OPEN_MP), every single ParallelFor call constructed
				// three std::function objects and an atomic, then destroyed them untouched.
				// The OpenMP branch calls _ChunkFunction directly and never looks at any of
				// them. Each std::function whose captures exceed the small-buffer size is a
				// heap allocation, and the solver issues on the order of 10^4-10^5
				// ParallelFor calls per run.
				//
				// Type erasure IS needed here -- the schedule is a runtime choice and both
				// arms must share a type -- but only once per call, and only on this branch,
				// which this build never takes. The chosen arm is assigned directly rather
				// than built as two intermediates and copied.
				std::atomic< size_t > index;
				index.store( 0 );
				std::function< void (unsigned int ) > ThreadFunction;
				if( schedule==ScheduleType::STATIC )
					ThreadFunction = [ &_ChunkFunction , chunks , numThreads ]( unsigned int thread )
						{
							for( size_t chunk=thread ; chunk<chunks ; chunk+=numThreads ) _ChunkFunction( thread , chunk );
						};
				else if( schedule==ScheduleType::DYNAMIC )
					ThreadFunction = [ &_ChunkFunction , chunks , &index ]( unsigned int thread )
						{
							size_t chunk;
							while( ( chunk=index.fetch_add(1) )<chunks ) _ChunkFunction( thread , chunk );
						};

				static std::vector< std::future< void > > futures;
				futures.resize( numThreads-1 );
				for( unsigned int t=1 ; t<numThreads ; t++ ) futures[t-1] = std::async( std::launch::async , ThreadFunction , t );
				ThreadFunction( 0 );
				for( unsigned int t=1 ; t<numThreads ; t++ ) futures[t-1].get();
			}
		}

	private:
		static unsigned int _NumThreads;

		template< typename Function , typename ... Functions >
		static void _ParallelSections( std::vector< std::future< void > > &futures , const Function &function , const Functions & ... functions )
		{
			SectionCount.fetch_add( 1 , std::memory_order_relaxed );
			futures.push_back( std::async( std::launch::async , function ) );
			if constexpr( sizeof...(Functions) ) _ParallelSections( futures , functions... );
		}

		template< typename Function , typename ... Functions >
		static void _ParallelSections( std::vector< std::future< void > > &futures , const Function &&function , const Functions && ... functions )
		{
			SectionCount.fetch_add( 1 , std::memory_order_relaxed );
			futures.push_back( std::async( std::launch::async , function ) );
			if constexpr( sizeof...(Functions) ) _ParallelSections( futures , std::move(functions)... );
		}
	};

	inline ThreadPool::ParallelType ThreadPool::ParallelizationType = ThreadPool::ParallelType::NONE;
	inline unsigned int ThreadPool::_NumThreads = std::thread::hardware_concurrency();
	inline std::atomic< size_t > ThreadPool::SectionCount{ 0 };
	inline ThreadPool::ScheduleType ThreadPool::Schedule = ThreadPool::DYNAMIC;
	inline size_t ThreadPool::ChunkSize = 128;
	// Conservative starting point: 2048 iterations against a ~5-20 us fork/join means
	// the cutoff only fires where even a 10 ns body could not repay the region. Raise it
	// while the coarse-depth "Solved in" times keep falling and depth 10 does not move;
	// back it off if a short loop with an expensive body starts showing up. 0 disables.
	//
	//
	// MEASURED ON MARCO ULISES (depth 12, 70.4M octree nodes, 53% ghost), cutoff 0 vs 2048:
	//   depth  5     35,937 nodes   0.321 s -> 0.004   80x
	//   depth  7    110,760         0.604   -> 0.013   46x
	//   depth  9  1,424,784         1.177   -> 0.325    3.6x
	//   depth 10  5,690,592         0.915   -> 1.043    none
	//   depth 11 18,095,208         1.820   -> 1.807    none
	//   depth 12  6,936,360        11.777   -> 1.415    8.3x   <-- the FINEST level
	//   linear solve 26.4 s -> 10.5 | extraction 21.5 -> 16.7 | surface stage 99.3 -> 77.8
	//
	// Note depth 12, which breaks the "only coarse levels are overhead-bound" reading this
	// was first justified with. A sparse elongated scene has few nodes per slice even at
	// its finest level, so the ranges stay small there too. Depths 10 and 11 are untouched
	// because those levels DO have dense slices. Range, not depth, is what decides.
	//
	// EXONERATED as the cause of PoissonRecon's "bad average roots" warning: at cutoff 0
	// the warning still fires, with a different count (12 against 27), so it is run-to-run
	// variation from the dynamic schedule and not this. It is also not new -- the earlier
	// baseline appeared clean only because a .log file captures the OpenMVS logger and not
	// raw stdout, and MK_WARN writes to stdout.
	// OPENMVS_POISSON_SERIAL_CUTOFF overrides, so this can be swept without a rebuild.
	//
	// 2048 was calibrated against CHEAP loop bodies, where a 32-thread region costs about
	// what 2048 trivial iterations do. The gate is on ITERATION COUNT ALONE, so it applies
	// that same number to expensive bodies too -- and those are exactly the loops worth
	// threading. The Gauss-Seidel relaxation dispatches ~373 items per colour at depth 11
	// with a row update costing 60-200 cycles: ~9 us of work, which a ~1-2 us OpenMP region
	// should still beat by several times, yet it runs serial.
	//
	// SWEPT ON HISTORIC, THREE POINTS, QUIET BOX. 2048 stands. "Solved in" per depth:
	//
	//   cutoff:         0      512     2048
	//   depth  5     0.322    0.113    0.004
	//   depth  7     0.311    0.195    0.026
	//   depth  9     0.283    0.165    0.195
	//   depth 10     0.752    0.528    0.505
	//   depth 11     1.517    1.016    1.075
	//   SOLVE TOTAL  5.6 s    3.8 s    3.6 s
	//   Got Faces    6.2 s      -      5.2 s
	//
	// 0 is the UPSTREAM behaviour -- this cutoff does not exist in Kazhdan's code, where
	// every ParallelFor opens a region however small. It costs ~2 s on the solve, every
	// depth worse, the coarse ones by ~80x. 65536 is also much slower (it serialises the
	// per-slice matrix assembly and the extractor's per-slice loops). Worse in both
	// directions is what a tuned threshold looks like.
	//
	// THE PER-CALL-SITE CUTOFF IS PRICED, AND IT IS NOT WORTH BUILDING. The 512 column
	// shows the effect is real: depths 9 and 11 ARE faster with a lower cutoff, because
	// their loop bodies are expensive enough to pay for a region. But it is worth 0.066 s
	// across depths 9-11, against 0.28 s lost at depths 5-8. A hypothetical per-depth
	// cutoff -- 2048 coarse, 512 fine -- therefore buys ~0.07 s of a 3.6 s solve: 2% of the
	// solve, 0.3% of the stage. ParallelFor already takes chunkSize and numThreads per
	// call, so the mechanism exists; the payoff does not justify using it.
	//
	// MEASUREMENT NOTE: an earlier 512 run showed Set Table at 12.1 s and Got Faces at
	// 16.1 s. It was machine load, not the cutoff -- cutoff 0 makes strictly more loops
	// parallel and shows Set Table at 0.25/0.16, identical to 2048. Check an unrelated
	// phase ("Read input into tree" is ideal: serial, and finishes before any of this)
	// before believing a point on this curve.
	//
	// 0 disables the cutoff entirely.
	inline size_t ThreadPool::SerialCutoff = []( void ) -> size_t
	{
		if( const char *e = std::getenv( "OPENMVS_POISSON_SERIAL_CUTOFF" ) )
		{
			const long long v = std::atoll( e );
			if( v>=0 ) return (size_t)v;
		}
		return 2048;
	}();

	const inline std::vector< std::string > ThreadPool::ParallelNames =
	{
#ifdef _OPENMP
		"open mp" ,
#endif // _OPENMP
		"async" ,
		"none"
	};
	const inline std::vector< std::string > ThreadPool::ScheduleNames = { "static" , "dynamic" };
}
#endif // MULTI_THREADING_INCLUDED
