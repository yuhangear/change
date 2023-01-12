#!/usr/bin/env python

from __future__ import division
import argparse
from ws4py.client.threadedclient import WebSocketClient
import time
import threading
import sys
import urllib.parse
import queue
import json
import time
import ssl
from datetime import datetime

CHANNELS = 1
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

import logging
# create logger
logger = logging.getLogger('client')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logfh = logging.handlers.RotatingFileHandler('client_wenet_short1.log', maxBytes=10485760, backupCount=10)
logfh.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(u'%(levelname)8s %(asctime)s %(message)s ')
logging._defaultFormatter = logging.Formatter(u"%(message)s")

# add formatter to ch
ch.setFormatter(formatter)
logfh.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(logfh)

def rate_limited(maxPerSecond):
    minInterval = 1.0 / float(maxPerSecond)
    def decorate(func):
        lastTimeCalled = [0.0]
        def rate_limited_function(*args,**kargs):
            elapsed = time.perf_counter() - lastTimeCalled[0]
            leftToWait = minInterval - elapsed
            if leftToWait>0:
                time.sleep(leftToWait)
            ret = func(*args,**kargs)
            lastTimeCalled[0] = time.perf_counter()
            return ret
        return rate_limited_function
    return decorate


class MyClient(WebSocketClient):

    def __init__(self, mode, audiofile, url, protocols=None, extensions=None, heartbeat_freq=None, byterate=32000,
                 save_adaptation_state_filename=None, ssl_options=None, send_adaptation_state_filename=None, tag="0"):
        super(MyClient, self).__init__(url, protocols, extensions, heartbeat_freq)
        self.final_hyps = []
        self.audiofile = audiofile
        self.byterate = byterate
        self.final_hyp_queue = queue.Queue()
        self.save_adaptation_state_filename = save_adaptation_state_filename
        self.send_adaptation_state_filename = send_adaptation_state_filename
        self.dt1 = datetime.now()
        self.ssl_options = ssl_options or {}
        self.total_len = 0
        if self.scheme == "wss":
            # Prevent check_hostname requires server_hostname (ref #187)
            if "cert_reqs" not in self.ssl_options:
                self.ssl_options["cert_reqs"] = ssl.CERT_NONE
        self.tag = tag
        self.mode = mode
        # self.audio = pyaudio.PyAudio()
        self.isStop = False

    @rate_limited(25)
    def send_data(self, data):
        self.send(data, binary=True)

    def opened(self):
        logger.info("Socket opened! " + self.__str__())

        start_signal = json.dumps({"signal": "start","continuous_decoding": True,"nbest":1, "sample_rate":16000, "hot_list":0})
                         
        self.send(start_signal)

    def send_data_to_ws(self):
        if self.send_adaptation_state_filename is not None:
            logger.info("Sending adaptation state from %s" % self.send_adaptation_state_filename)
            try:
                adaptation_state_props = json.load(open(self.send_adaptation_state_filename, "r"))
                self.send(json.dumps(dict(adaptation_state=adaptation_state_props)))
            except:
                e = sys.exc_info()[0]
                logger.info("Failed to send adaptation state: %s" % e)

        logger.info("Start transcribing...")
        if self.mode == 'stream':
            logger.info("Server doesn't have sound card, skip it")
            # stream = self.audio.open(format=FORMAT, channels=CHANNELS,
            #     rate=RATE, input=True,
            #     frames_per_buffer=CHUNK)
            # while not self.isStop:
            #     data = stream.read(int(self.byterate / 8), exception_on_overflow=False)
            #     self.send_data(data) # send data
            #
            # stream.stop_stream()
            # stream.close()
            # self.audio.terminate()
        elif self.mode == 'file':
            logger.info("Starting to transcript the file:%s"%(self.audiofile.name))
            with self.audiofile as audiostream:
                self.dt1 = datetime.now()
                for block in iter(lambda: audiostream.read(1280), ""):
                    self.total_len = self.total_len + int(len(block)/2)
                    #logger.info("Send %i data"%(int(len(block)/2)))
                    if len(block)==0:
                        break
                    self.send_data(block)
                    self.dt3 = datetime.now()
                    

        logger.info("Audio sent, now sending EOS")
        self.send("EOS")


    def received_message(self, m):
        response = json.loads(str(m))
        #print(response)
        if response['status'] == 200:
            t = threading.Thread(target=self.send_data_to_ws)
            t.start()
        
        # if response['status'] == 'ok':
        #     if response["type"] == 'server_ready':
        #         t = threading.Thread(target=self.send_data_to_ws)
        #         t.start()
        #     else:
        #         print(response)

        #if response["type"] == "final_result":
        #    self.dt2 = datetime.now()
        #    delta = self.dt2 - self.dt1
        #    trans = json.loads(response)
        #    self.final_hyps.append(trans)
        #    logger.info('+%i s: %s' % (delta.seconds, trans))
                
        if response['status'] == 0:
            if 'result' in response:
                #trans = response['result']['hypotheses']
                self.dt2 = datetime.now()
                delta = self.dt2 - self.dt1
                trans = response['result']['hypotheses'][0]['transcript']
                
                if response['result']['final']:
                     #print >> sys.stderr, trans,
                     self.final_hyps.append(trans)
                     #print("\033[H\033[J") # clear console for better output
                     if trans != "":
                        logger.info('[final] +%f: %s' %((delta.seconds+delta.microseconds/1000000),trans))
                else:
                     print_trans = trans
                     if len(print_trans) > 80:
                         print_trans = "... %s" % print_trans[-76:]

                     #print("\033[H\033[J") # clear console for better output
                     #logger.info('[partial] +%f: %s' % ((delta.seconds+delta.microseconds/1000000),trans))
            elif response['type'] == "speech_end":
                latency = int((self.dt2 - self.dt3).total_seconds()*1000)
                decode_taken = int((self.dt2 - self.dt1).total_seconds()*1000)
                audio_len = int(self.total_len/16000*1000)
                rtf = decode_taken/audio_len
                logger.info("[%s] Total latency: %d ms"%(self.tag, latency))
                logger.info("[%s] Decoded %dms audio taken %dms"%(self.tag, audio_len,decode_taken))
                logger.info("[%s] RTF: %f "%(self.tag, rtf))
                self.dt1 = datetime.now()
        #     if 'adaptation_state' in response:
        #         if self.save_adaptation_state_filename:
        #             logger.info("Saving adaptation state to %s" % self.save_adaptation_state_filename)
        #             with open(self.save_adaptation_state_filename, "w") as f:
        #                 f.write(json.dumps(response['adaptation_state']))
        # else:
        #     logger.info("Received error from server (status %d)" % response['status'])
        #     if 'message' in response:
        #         logger.info("Error message: %s" %  response['message'])


    def get_full_hyp(self, timeout=60):
        return self.final_hyp_queue.get(timeout)

    def closed(self, code, reason=None):
        #print "Websocket closed() called"
        #print >> sys.stderr
        self.final_hyp_queue.put(" ".join(self.final_hyps))


def main():

    parser = argparse.ArgumentParser(description='Command line client for kaldigstserver')
    parser.add_argument('-o', '--option', default="file", dest="mode", help="Mode of transcribing: audio file or streaming")
    parser.add_argument('-u', '--uri', default="ws://localhost:8888/client/ws/speech", dest="uri", help="Server websocket URI")
    parser.add_argument('-r', '--rate', default=32000, dest="rate", type=int, help="Rate in bytes/sec at which audio should be sent to the server. NB! For raw 16-bit audio it must be 2*samplerate!")
    parser.add_argument('-t', '--token', default="", dest="token", help="User token")
    parser.add_argument('-m', '--model', default=None, dest="model", help="model in azure container")
    parser.add_argument('-l', '--log', default="test_wenet1.log", dest="log", help="save the text result")
    parser.add_argument('--tag', default=0, dest="tag", help="tag")
    parser.add_argument('--save-adaptation-state', help="Save adaptation state to file")
    parser.add_argument('--send-adaptation-state', help="Send adaptation state from file")
    parser.add_argument('--content-type', default='', help="Use the specified content type (empty by default, for raw files the default is  audio/x-raw, layout=(string)interleaved, rate=(int)<rate>, format=(string)S16LE, channels=(int)1")
    parser.add_argument('audiofile', nargs='?', help="Audio file to be sent to the server", type=argparse.FileType('rb'), default=sys.stdin)
    args = parser.parse_args()
    if args.mode == 'file' or args.mode == 'stream':
        content_type = args.content_type
        if content_type == '' and args.audiofile.name.endswith(".raw") or args.mode == 'stream':
            content_type = "audio/x-raw, layout=(string)interleaved, rate=(int)%d, format=(string)S16LE, channels=(int)1" %(args.rate/2)

        ws = MyClient(args.mode, args.audiofile, args.uri + '?%s' % (urllib.parse.urlencode([("content-type", content_type)])) + '&%s' % (urllib.parse.urlencode([("token", args.token)])) + '&%s' % (urllib.parse.urlencode([("token", args.token)])) + '&%s' % (urllib.parse.urlencode([("model", args.model)])), byterate=args.rate,
                    save_adaptation_state_filename=args.save_adaptation_state, send_adaptation_state_filename=args.send_adaptation_state, tag = args.tag)


        ws.connect()
        result = ws.get_full_hyp()

        with open(args.log,'a',encoding='utf-8') as file:
            file.write(str(args.audiofile.name.split("/")[-1].replace(".wav","")) + " " +result + '\n')
            file.close()

        logger.info("\n URL: " + str(args.uri + '?%s' % (urllib.parse.urlencode([("content-type", content_type)])) + '?%s' % (urllib.parse.urlencode([("token", args.token)]))) + "\n")
        logger.info("[%s] Final Result: %s "%(args.tag, result))
    else:
        print('\nTranscribe mode must be file or stream!\n')

if __name__ == "__main__":
    main()
