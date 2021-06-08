#ifndef _STOPWATCH_H_INCLUDED_
#define _STOPWATCH_H_INCLUDED_

#include <time.h>

#ifdef USESTOPWATCH
// accurate linux timing
class stopwatch
{
private:
   timespec start;
   timespec stop;
public:
   void tic();
   double toc();
};

timespec diff(timespec start, timespec end);

inline void stopwatch::tic()
{
   clock_gettime(CLOCK_MONOTONIC, &start);
}
inline double stopwatch::toc()
{
   clock_gettime(CLOCK_MONOTONIC, &stop);
   return (diff(start,stop).tv_sec) + (double)(diff(start,stop).tv_nsec)*1e-9;
}

// calculate the difference in times
timespec diff(timespec start, timespec end)
{
   timespec ts;
   ts.tv_sec = end.tv_sec - start.tv_sec;
   ts.tv_nsec = end.tv_nsec - start.tv_nsec;
   if (ts.tv_nsec < 0) {
      ts.tv_sec--;
      ts.tv_nsec += 1000000000;
   }
   return ts;
}

#else
// just count iterations
class stopwatch
{
public:
   void tic();
   double toc();
};

inline void stopwatch::tic()
{
}
inline double stopwatch::toc()
{
   return 1;
}

#endif

#endif
