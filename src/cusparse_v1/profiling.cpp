
#ifdef profile
#include <time.h>
#endif

#ifndef __PROFILE_H__
#define __PROFILE_H__
#include "profiling.h"
#endif

#include <iostream>
int initOK = 0;
std::ofstream timingFile;

void profile_start(timespec *timeStart, int profileFlag)
{

  clock_gettime(CLOCK_MONOTONIC, timeStart);

}

double profile_end(timespec *timeStart, timespec *timeEnd)
{
  double timediff = 0.0;
  timespec temp;
  clock_gettime(CLOCK_MONOTONIC, timeEnd);
  if ((timeEnd->tv_nsec-timeStart->tv_nsec)<0) 
    {
      temp.tv_sec = timeEnd->tv_sec-timeStart->tv_sec-1;
      temp.tv_nsec = 1000000000+timeEnd->tv_nsec-timeStart->tv_nsec;
    } 
  else 
    {
      temp.tv_sec = timeEnd->tv_sec-timeStart->tv_sec;
      temp.tv_nsec = timeEnd->tv_nsec-timeStart->tv_nsec;
    }	
  timediff = (temp.tv_nsec) / 1e9;
  timediff = timediff + temp.tv_sec;
  return timediff;    
}

int profile_init(char *logFile)
{
  #ifdef profile
  char dateStr[100];    
  time_t T = time(NULL);
  struct tm ltime = *localtime(&T);
  sprintf(dateStr, "%s_%d_%d_%d_%d_%d_%d",logFile, ltime.tm_mday, ltime.tm_mon + 1, ltime.tm_year + 1900, ltime.tm_hour, ltime.tm_min, ltime.tm_sec);
  timingFile.open(dateStr);
  if(timingFile.is_open())
    {
      initOK = 1;
      printf("Log file created\n");
    }
  else
    {
      printf("Could not create log file created\n");
    }
  return initOK;
  
  #else
  return 1;
  #endif
}


