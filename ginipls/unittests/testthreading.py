#!/usr/bin/python

import queue
import threading
import time

exitFlag = 0

class myThread (threading.Thread):
    def __init__(self, threadID, name, q, queueLock):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.q = q
      self.queueLock = queueLock
    def run(self):
      print("Starting %s" % self.name)
      self.process_data()
      print("Exiting %s" % self.name)

    def process_data(self):
        while not exitFlag:
            self.queueLock.acquire()
            if not self.q.empty():
                data = self.q.get()
                self.queueLock.release()
                print("%s processing %s" % (self.name, data['n']))
                data['t'] = self.name
            else:
                self.queueLock.release()
            time.sleep(1)

threadList = ["Thread-1", "Thread-2", "Thread-3"]
nameList = [{'n':"One"}, {'n':"Two"}, {'n':"Three"}, {'n':"Four"}, {'n':"Five"}]
queueLock = threading.Lock()
workQueue = queue.Queue(10)
threads = []
threadID = 1
# Create new threads
for tName in threadList:
   thread = myThread(threadID, tName, workQueue, queueLock)
   thread.start()
   threads.append(thread)
   threadID += 1

# Fill the queue
queueLock.acquire()
for word in nameList:
   workQueue.put(word)
queueLock.release()

# Wait for queue to empty
while not workQueue.empty():
   pass

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
   t.join()
print("Exiting Main Thread : %s " % str(nameList))