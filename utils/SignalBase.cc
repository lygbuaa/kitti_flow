/* SignalBase.cc */
#include "SignalBase.h"

void SignalBase::CatchSignal(){
    signal(SIGABRT, QuitHandler);// 6
    signal(SIGTERM, QuitHandler);// 15

    /* catch segment fault */
    struct sigaction sa;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = SigSegvHandler;
    sa.sa_flags   = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, NULL);
}

void SignalBase::QuitHandler(int sig){
    LOG(ERROR) << "@q@ (pid " << getpid() << ", catch quit signal: " << sig;
    LOG(ERROR) << "call-stack dumped:\n" << LogFatalException::BackTrace(8, 2);
    //use _exit(), exit() may cause re-enter problem
    _exit(sig);
}

void SignalBase::SigSegvHandler(int sig, siginfo_t *si, void *arg){
    LOG(ERROR) << "@q@ (pid " << getpid() << ", segment fault @: " << si -> si_addr;
    LOG(ERROR) << "call-stack dumped:\n" << LogFatalException::BackTrace(8, 2);
    //use _exit(), exit() may cause re-enter problem
    _exit(sig);
}