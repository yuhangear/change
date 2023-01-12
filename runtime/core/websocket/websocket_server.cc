// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "websocket/websocket_server.h"
// #include "decoder/params.h"
#include <thread>
#include <utility>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "boost/json/src.hpp"
#include "utils/log.h"
#include "utils/string.h"
#include "frontend/resample.h"
#include <Python.h>

#include <memory>
#include <string>

#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>

#include <dirent.h>
#include <cstddef>







void getFiles(std::string path, std::vector<std::string> &files)
{
  DIR *dir;
  struct dirent *ptr;

  if ((dir = opendir(path.c_str())) == NULL)
  {
    perror("Open dir error...");
    exit(1);
  }

  /*
   * 文件(8)、目录(4)、链接文件(10)
   */

  while ((ptr = readdir(dir)) != NULL)
  {
    if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
      continue;
    else if (ptr->d_type == 8)
      files.push_back(path + ptr->d_name);
    else if (ptr->d_type == 10)
      continue;
    else if (ptr->d_type == 4)
    {
      //files.push_back(ptr->d_name);
      getFiles(path + ptr->d_name + "/", files);
    }
  }
  closedir(dir);
}



namespace wenet {




namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http = beast::http;            // from <boost/beast/http.hpp>
namespace websocket = beast::websocket;  // from <boost/beast/websocket.hpp>
namespace asio = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;        // from <boost/asio/ip/tcp.hpp>
namespace json = boost::json;

ConnectionHandler::ConnectionHandler(
    tcp::socket&& socket, std::shared_ptr<FeaturePipelineConfig> feature_config,
    std::shared_ptr<DecodeOptions> decode_config,
    std::shared_ptr<DecodeResource> decode_resource)
    : ws_(std::move(socket)),
      feature_config_(std::move(feature_config)),
      decode_config_(std::move(decode_config)),
      decode_resource_(std::move(decode_resource)) {}

void ConnectionHandler::OnSpeechStart() {
  LOG(INFO) << "Received speech start signal, start reading speech";
  got_start_tag_ = true;
  // json::value rv = {{"status", "ok"}, {"type", "server_ready"}};
  json::value rv = {{"status", 200}, {"type", "server_ready"}};
  ws_.text(true);
  ws_.write(asio::buffer(json::serialize(rv)));
  feature_pipeline_ = std::make_shared<FeaturePipeline>(*feature_config_);
  
  if (hot_list==0){
      decode_resource_->context_graph= nullptr;
      LOG(INFO) << "不使用热词解码 " ;
  }
  else if (hot_list>0 &&  hot_list<=99 && (decode_resource_->context_graph_list[hot_list-1])->if_init==1 ){
      LOG(INFO) << "热词 if_init " << (decode_resource_->context_graph_list[hot_list-1])->if_init ;
      decode_resource_->context_graph = decode_resource_->context_graph_list[hot_list-1];
      LOG(INFO) << "使用第"<<hot_list<<"个列表解码 " ;
  }
  else{
      decode_resource_->context_graph= nullptr;
      LOG(INFO) << "不存在指定热词列表,不使用热词解码 " ;

  }
  decoder_ = std::make_shared<AsrDecoder>(feature_pipeline_, decode_resource_,
                                          *decode_config_);
  // Start decoder thread
  decode_thread_ =
      std::make_shared<std::thread>(&ConnectionHandler::DecodeThreadFunc, this);
}

void ConnectionHandler::OnSpeechEnd() {
  LOG(INFO) << "Received speech end signal";
  if (feature_pipeline_ != nullptr) {
    feature_pipeline_->set_input_finished();
  }
  got_end_tag_ = true;
}

void ConnectionHandler::OnPartialResult(const std::string& result) {
  LOG(INFO) << "Partial result: " << result;
  // json::value rv = {
  //    {"status", "ok"}, {"type", "partial_result"}, {"nbest", result}};
  json::value v = json::parse(result);
  json::object jresult = {{"hypotheses", v}, {"final", false}};
  json::value rv = {{"status", 0}, {"id", 0}, {"result", jresult}};
  ws_.text(true);
  ws_.write(asio::buffer(json::serialize(rv)));
}

void ConnectionHandler::OnFinalResult(const std::string& result) {
  LOG(INFO) << "Final result: " << result;
  // json::value rv = {
  //    {"status", "ok"}, {"type", "final_result"}, {"nbest", result}};
  json::value v = json::parse(result);
  json::value total_length = v.get_array()[0].get_object()["segment-end"];
  json::value segment_start = v.get_array()[0].get_object()["segment-start"];
  json::value segment_len = v.get_array()[0].get_object()["segment-length"];
  json::object jresult = {{"hypotheses", v}, {"final", true}};
  json::value rv = {{"status", 0},
                    {"id", 0},
                    {"result", jresult},
                    {"segment-length", segment_len},
                    {"segment-start", segment_start},
                    {"total-length", total_length}};
  ws_.text(true);
  ws_.write(asio::buffer(json::serialize(rv)));
}

void ConnectionHandler::OnFinish() {
  // Send finish tag
  // json::value rv = {{"status", "ok"}, {"type", "speech_end"}};
  json::value rv = {{"status", 0}, {"type", "speech_end"}};
  ws_.text(true);
  ws_.write(asio::buffer(json::serialize(rv)));
}

//在这里修改采用率
void ConnectionHandler::OnSpeechData(const beast::flat_buffer& buffer) {
  // Read binary PCM data

  int num_samples = buffer.size() / sizeof(int16_t);
  VLOG(2) << "Received " << num_samples << " samples";
  CHECK(feature_pipeline_ != nullptr);
  CHECK(decoder_ != nullptr);
  const auto* pcm_data = static_cast<const int16_t*>(buffer.data().data());

  vector<float> data_change1(pcm_data, pcm_data+num_samples);
  vector<float>  data_changed;
  //need to set flag
  // 
  // if(num_samples==8000 || (num_samples!=4000&& num_samples%2==0) ){
  //   resample<float> ( 1, 1, data_change1, data_changed );
  //   feature_pipeline_->AcceptWaveform(&data_changed[0], num_samples);
  // }else {
  //   //4000  8k
  //   resample<float> ( 2, 1, data_change1, data_changed );
  //   feature_pipeline_->AcceptWaveform(&data_changed[0], num_samples*2);
  // }

  //sample_rate=(unsigned int)(sample_rate/1000);
  unsigned int  FenMu =(unsigned int)(sample_rate/1000);
  unsigned int FenZi =16;
  unsigned int GongYue= 1; //公约数
	
	for(int i=1;i<=FenZi;i++)         //从1开始逐个找公约数
	{
		if((FenZi%i==0) && (FenMu%i==0))     //看i是否为公约数
		{
			if(i>GongYue)               //看i是否比i之前的最大公约数要大
			{
				GongYue=i;              
			}
			continue;
		}
		continue;                        //不是的话就跳过看下一个数
	} 
	
	FenZi=(unsigned int)(FenZi/GongYue);               //求分子化简后数字
	FenMu=(unsigned int)(FenMu/GongYue);               //求分母化简后数字
  unsigned int beishu = (unsigned int)(FenZi/FenMu);

  resample<float> ( FenZi, FenMu, data_change1, data_changed );
  feature_pipeline_->AcceptWaveform(&data_changed[0], num_samples*beishu);

  // if(sample_rate==16000 ){
  //   resample<float> ( 1, 1, data_change1, data_changed );
  //   feature_pipeline_->AcceptWaveform(&data_changed[0], num_samples);
  // }else {
  //   //4000  8k
  //   resample<float> ( 2, 1, data_change1, data_changed );
  //   feature_pipeline_->AcceptWaveform(&data_changed[0], num_samples*2);
  // }
  


  
}

  /* 
std::string ConnectionHandler::SerializeResult(bool finish) {
  json::array nbest;
  for (const DecodeResult& path : decoder_->result()) {
    json::object jpath({{"sentence", path.sentence}});
    if (finish) {
      json::array word_pieces;
      for (const WordPiece& word_piece : path.word_pieces) {
        json::object jword_piece({{"word", word_piece.word},
                                  {"start", word_piece.start},
                                  {"end", word_piece.end}});
        word_pieces.emplace_back(jword_piece);
      }
      jpath.emplace("word_pieces", word_pieces);
    }
    nbest.emplace_back(jpath);

    if (nbest.size() == nbest_) {
      break;
    }
  }
  return json::serialize(nbest);
}
  */

std::string ConnectionHandler::SerializeResult(bool finish) {
  json::array nbest;
  for (const DecodeResult& path : decoder_->result()) {
    // json::object jpath({{"sentence", path.sentence}});
    json::object jpath({{"transcript", path.sentence}});
    if (finish) {
      json::array word_pieces;
      int segment_end;

      if (path.word_pieces.size() != 0) {
        int segment_start = path.word_pieces.at(0).start;
        for (const WordPiece& word_piece : path.word_pieces) {
          json::object jword_piece({{"word", word_piece.word},
                                    {"start", word_piece.start},
                                    {"end", word_piece.end}});
          word_pieces.emplace_back(jword_piece);
          segment_end = word_piece.end;
        }
        // jpath.emplace("word_pieces", word_pieces);
        jpath.emplace("word-alignment", word_pieces);

        jpath.emplace("segment-end", segment_end / 1000.0);
        jpath.emplace("segment-start", segment_start / 1000.0);
        jpath.emplace("segment-length", (segment_end - segment_start) / 1000.0);
      } else {
        jpath.emplace("segment-end", 0);
        jpath.emplace("segment-start", 0);
        jpath.emplace("segment-length", 0);
      }
    }
    nbest.emplace_back(jpath);

    if (nbest.size() == nbest_) {
      break;
    }
  }
  return json::serialize(nbest);
}
  
void ConnectionHandler::DecodeThreadFunc() {
  try {
    while (true) {
      DecodeState state = decoder_->Decode();
      if (state == DecodeState::kEndFeats) {
        decoder_->Rescoring();
        std::string result = SerializeResult(true);
        OnFinalResult(result);
        OnFinish();
        stop_recognition_ = true;
        break;
      } else if (state == DecodeState::kEndpoint) {
        decoder_->Rescoring();
        std::string result = SerializeResult(true);
        OnFinalResult(result);
        // If it's not continuous decoding, continue to do next recognition
        // otherwise stop the recognition
        if (continuous_decoding_) {
          decoder_->ResetContinuousDecoding();
        } else {
          OnFinish();
          stop_recognition_ = true;
          break;
        }
      } else {
        if (decoder_->DecodedSomething()) {
          std::string result = SerializeResult(false);
          OnPartialResult(result);
        }
      }
    }
  } catch (std::exception const& e) {
    LOG(ERROR) << e.what();
  }
}

void ConnectionHandler::OnError(const std::string& message) {
  json::value rv = {{"status", "failed"}, {"message", message}};
  ws_.text(true);
  ws_.write(asio::buffer(json::serialize(rv)));
  // Close websocket
  ws_.close(websocket::close_code::normal);
}
  /*
void ConnectionHandler::OnText(const std::string& message) {
  json::value v = json::parse(message);
  if (v.is_object()) {
    json::object obj = v.get_object();
    if (obj.find("signal") != obj.end()) {
      json::string signal = obj["signal"].as_string();
      if (signal == "start") {
        if (obj.find("nbest") != obj.end()) {
          if (obj["nbest"].is_int64()) {
            nbest_ = obj["nbest"].as_int64();
          } else {
            OnError("integer is expected for nbest option");
          }
        }
        if (obj.find("continuous_decoding") != obj.end()) {
          if (obj["continuous_decoding"].is_bool()) {
            continuous_decoding_ = obj["continuous_decoding"].as_bool();
          } else {
            OnError(
                "boolean true or false is expected for "
                "continuous_decoding option");
          }
        }
        if (obj.find("hot_list") != obj.end()) {
          if (obj["hot_list"].is_int64()) {
            hot_list = obj["hot_list"].as_int64();
          } else {
            OnError("integer is expected for nbest option");
          }
        }
        if (obj.find("sample_rate") != obj.end()) {
          if (obj["sample_rate"].is_int64()) {
            sample_rate = obj["sample_rate"].as_int64();
          } else {
            OnError("integer is expected for nbest option");
          }
        }
        OnSpeechStart();
      } else if (signal == "end") {
        OnSpeechEnd();
      } else {
        OnError("Unexpected signal type");
      }
    } else {
      OnError("Wrong message header");
    }
  } else {
    OnError("Wrong protocol");
  }
}

  */

void ConnectionHandler::OnText(const std::string& message) {
  if (message == "EOS") {
    OnSpeechEnd();
  } else if (message == "keepalive") {
    LOG(INFO) << "Received keepalive";
  } else {
    json::value v = json::parse(message);
    if (v.is_object()) {
      json::object obj = v.get_object();
      if (obj.find("signal") != obj.end()) {
        json::string signal = obj["signal"].as_string();
        if (signal == "start") {
          if (obj.find("nbest") != obj.end()) {
            if (obj["nbest"].is_int64()) {
              nbest_ = obj["nbest"].as_int64();
            } else {
              OnError("integer is expected for nbest option");
            }
          }
          if (obj.find("continuous_decoding") != obj.end()) {
            if (obj["continuous_decoding"].is_bool()) {
              continuous_decoding_ = obj["continuous_decoding"].as_bool();
            } else {
              OnError(
                  "boolean true or false is expected for "
                  "continuous_decoding option");
            }
          }
          if (obj.find("hot_list") != obj.end()) {
            if (obj["hot_list"].is_int64()) {
              hot_list = obj["hot_list"].as_int64();
            } else {
              OnError("integer is expected for nbest option");
            }
          }
          if (obj.find("sample_rate") != obj.end()) {
            if (obj["sample_rate"].is_int64()) {
              sample_rate = obj["sample_rate"].as_int64();
            } else {
              OnError("integer is expected for nbest option");
            }
          }
          OnSpeechStart();
        } else if (signal == "end") {
          OnSpeechEnd();
        } else {
          OnError("Unexpected signal type");
        }
      } else {
        OnError("Wrong message header");
      }
    } else {
      OnError("Wrong protocol");
    }
  }
}
  
void ConnectionHandler::operator()() {
  try {
    // Accept the websocket handshake
    ws_.accept();
    for (;;) {
      // This buffer will hold the incoming message
      beast::flat_buffer buffer;
      // Read a message
      ws_.read(buffer);
      if (ws_.got_text()) {
        std::string message = beast::buffers_to_string(buffer.data());
        LOG(INFO) << message;
        OnText(message);
        if (got_end_tag_) {
          break;
        }
      } else {
        if (!got_start_tag_) {
          OnError("Start signal is expected before binary data");
        } else {
          if (stop_recognition_) {
            break;
          }
          OnSpeechData(buffer);
        }
      }
    }

    LOG(INFO) << "Read all pcm data, wait for decoding thread";
    if (decode_thread_ != nullptr) {
      decode_thread_->join();
    }
  } catch (beast::system_error const& se) {
    LOG(INFO) << se.code().message();
    // This indicates that the session was closed
    if (se.code() == websocket::error::closed) {
      OnSpeechEnd();
    }
    if (decode_thread_ != nullptr) {
      decode_thread_->join();
    }
  } catch (std::exception const& e) {
    LOG(ERROR) << e.what();
  }
}

void WebSocketServer::Start() {
  try {
    auto const address = asio::ip::make_address("0.0.0.0");
    tcp::acceptor acceptor{ioc_, {address, static_cast<uint16_t>(port_)}};





    struct stat hw_filestats_check[100];
    struct stat hw_filestats_check_old[100];

    std::vector<std::string> hot_word_list;
    std::vector<std::string> hot_word_list2;
    std::string context_path_old=decode_resource_->context_path ;
    context_path_old.pop_back();
    context_path_old=context_path_old+ "_old/";
    // getFiles(decode_resource_->context_path, hot_word_list);
    getFiles(context_path_old, hot_word_list);
    int i=0;
    for(auto &s : hot_word_list)
    {
      int i_1;
      i_1=s[s.size() - 2] - '0'  ;
      i=s[s.size() - 1] - '0' -1 ;
      if (i_1<=9 && i_1>=0){
        i=i_1*10+i;
      }
      else{
        i=s[s.size() - 1] - '0' -1 ;
      }
      if (lstat(s.c_str(), &hw_filestats_check_old[i]) == 0) {

        }

   
    }

    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("import sentencepiece as spm");
    PyRun_SimpleString("sp = spm.SentencePieceProcessor()");
    PyRun_SimpleString("import math");


    std::string bpe_path1 = "sys.path.append('" ;
    std::string bpe_path2= decode_resource_->bpe_path;
    std::string bpe_path3 ="')" ;
    std::string bpe_path=bpe_path1+ bpe_path2+bpe_path3;
    PyRun_SimpleString(bpe_path.c_str());
    LOG(INFO) << "bpe_path" <<bpe_path.c_str();
    PyRun_SimpleString("import bpe_cut");  // 导入py文件
    //PyRun_SimpleString("bpe_cut.cut('/datadrive/wenet/wenet-hotwords/temp/wenet/wenet/runtime/libtorch/hot_word_dir_old/hot1','/datadrive/wenet/wenet-hotwords/temp/wenet/wenet/runtime/libtorch/hot_word_dir/hot3')");  // 调用python函数
    // struct stat hw_filestats_check;
    // struct stat hw_filestats_check_old;
    // if (lstat((decode_resource_->context_path).c_str(), &hw_filestats_check_old) == 0) {

    //   }

    std::string context_path="";
    std::string command="";
    std::string command1="bpe_cut.cut('";
    std::string command2="','";
    std::string command3="')";
    command=command1+command2;
    for (;;) {
    getFiles(context_path_old, hot_word_list2);
      // This will receive the new connection
      tcp::socket socket{ioc_};
      // Block until we get a connection
      acceptor.accept(socket);
    int i =0;
    for(auto &s : hot_word_list2)
    {
      int i_1;
      i_1=s[s.size() - 2] - '0'  ;
      i=s[s.size() - 1] - '0' -1 ;
      if (i_1<=9 && i_1>=0){
        i=i_1*10+i;
      }
      else{
        i=s[s.size() - 1] - '0' -1 ;
      }
      
      if (lstat((s).c_str(), &hw_filestats_check[i]) == 0) {

            if ((hw_filestats_check[i].st_size != hw_filestats_check_old[i].st_size) && (hw_filestats_check[i].st_mtime != hw_filestats_check_old[i].st_mtime)) {
            //the file has been changed reload
            hw_filestats_check_old[i] = hw_filestats_check[i];
            //重新生成热词列表，原来的，通过bpe model
            context_path=s;

            
            if (i_1<=9 && i_1>=0){
            context_path.erase(context_path.size()-10,4);
            }
            else{
              
            context_path.erase(context_path.size()-9,4);
            }
            command= command1 + s + command2 + context_path + command3 ;
            LOG(INFO) << command.c_str() ;
            //bpe_cut.cut('./hot_word_dir_old/hot39','./hot_word_dir_hot39')
            //bpe_cut.cut('./hot_word_dir_old/hot9','./hot_word_dir/hot9')
            PyRun_SimpleString(command.c_str());  // 调用python函数




            LOG(INFO) << "context list is changed reloading from" << context_path;
            std::vector<std::string> contexts;
            std::ifstream infile(context_path);
            std::string context;
            while (getline(infile, context)) {
              contexts.emplace_back(Trim(context));
            }
            decode_resource_->context_graph_list[i]->BuildContextGraph(contexts, decode_resource_->symbol_table);
            decode_resource_->context_graph_list[i]->if_init==1 ;
            } else {
          LOG(INFO) << "context list is not changed";
              }
            }



    
    }

    hot_word_list2.clear();


      // decode_resource_->context_graph=decode_resource_->context_graph_list1;
      // Launch the session, transferring ownership of the socket
      ConnectionHandler handler(std::move(socket), feature_config_,
                                decode_config_, decode_resource_);
      std::thread t(std::move(handler));
      t.detach();
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << e.what();
  }
}

}  // namespace wenet
