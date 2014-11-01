#include <stdlib.h>
#include <stdio.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/unistd.h>

int main(int argc, char *argv[])
{
    char hostname[128];
    gethostname(hostname, sizeof hostname);
    printf("I am process with id %d running on %s core : %d\n", getpid(), hostname, sched_getcpu());
    return 0;
}
