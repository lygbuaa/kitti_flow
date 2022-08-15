/* SignalBase.h */
#ifndef SIGNAL_BASE_H
#define SIGNAL_BASE_H

#include <memory>
#include <unistd.h>
#include <signal.h>
#include <glog/logging.h>
#include "LogFatalException.h"

/*T is the derived-class name*/
class SignalBase
{
public:
    SignalBase(){};
    ~SignalBase(){};
    static void QuitHandler(int sig);
    static void SigSegvHandler(int sig, siginfo_t *si, void *arg);
    static void CatchSignal();
};

#endif //SIGNAL_BASE_H