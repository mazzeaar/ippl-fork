#include "../parallel_construction/algo01.hpp"
#include "../parallel_construction/algo02.hpp"
#include "../parallel_construction/algo03.hpp"
#include "../parallel_construction/algo04.hpp"
#include "../parallel_construction/algo05.hpp"
#include "../parallel_construction/algo08.hpp"


#include <Utility/IpplTimings.h>

void test_algo01() {
    // create a timer, or get one that already exists
    IpplTimings::TimerRef t = IpplTimings::getTimer("algo01");
    // start a timer
    IpplTimings::startTimer(t);
    // stop a timer, and accumulate it's values
    IpplTimings::stopTimer(t);
    // clear a timer, by turning it off and throwing away its time
    IpplTimings::clearTimer(t);
    // return a TimerInfo struct by asking for the name
    IpplTimings::infoTimer("algo01");
    // print the results to standard out
    IpplTimings::print();
}