

#include <time.h>
#include <fstream>
#include <string.h>

using namespace std;


void profile_start(timespec *timeStart, int profileFlag = 1);
double profile_end(timespec *timeStart, timespec *timeEnd);
int profile_init(char *logFile);
