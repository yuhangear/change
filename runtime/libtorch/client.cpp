
// g++ client.cpp -lpthread -o client


// usage:
// ./client upload hot1 
// ./client download hot1 


#include <netdb.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <sys/socket.h> 
#include <netinet/in.h> 
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <gflags/gflags.h>
#include <string>
#include <pthread.h> //necessary to work with threads
#include <fstream>
#include <semaphore.h> //this library will allow us to work with multiple process and thrreads
#define MAX 1024
#define PORT 5001
#define SA struct sockaddr 
using namespace std;

//initialization of int's we will be using through the code
int file_send_id=0;
int pause_id=0;
int resume_id=0;
int abort_id=0;
int quit=0;
int thread_count=0;
sem_t mutex;

DEFINE_string(port, "127.0.0.1", "the server host port");
DEFINE_string(type , "upload ", "upload or download");
DEFINE_string(file, "hot", "file name ");

struct parameter
{
        string argv1 ;
		string argv2 ;

};
string getMd5sum(string path)
{
	string execute = "md5sum ";
	execute.append(path);
	execute.append(" > md.txt");
	system(execute.c_str());
	ifstream FD;
	FD.open("md.txt");
	string readstr;
	getline(FD,readstr,' ');
	//cout<<readstr<<endl;
	return readstr;
}
void * send_file(void * param)
{
	string argv2 = *(string*)param;
	//structures declaration
	int sockfd, connfd; 
	struct sockaddr_in servaddr, cli; 

	// socket create and varification 
	sockfd = socket(AF_INET, SOCK_STREAM, 0); 
	if (sockfd == -1) { 
		printf("socket creation failed...\n"); 
		exit(0); 
	} 
	else
		printf("Socket successfully created..\n"); 
	bzero(&servaddr, sizeof(servaddr)); 

	// Here we assign IP and PORT to start processing the file holding
	servaddr.sin_family = AF_INET; 
	servaddr.sin_addr.s_addr = inet_addr("127.0.0.1"); 
	servaddr.sin_port = htons(PORT); 

	// connect the client socket to server socket 
	if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr)) != 0) { 
		printf("connection with the server failed...\n"); 
		exit(0); 
	} 
			//Initialize 
			char buff[MAX]; 
			int n; 
			fstream FD;
			string fromFile, Name, toFile, choice;
			
			strcpy(buff,"send\n");
			write(sockfd, buff, sizeof(buff)); 
			bzero(buff, sizeof(buff)); 
		
			n = 0; 
			/*while ((buff[n++] = getchar()) != '\n'); 
			write(sockfd, buff, sizeof(buff)); //send filename
			Name = buff;*/
			Name=argv2;
			cout << "File name is : " << Name;
			strcpy(buff,Name.c_str());
			write(sockfd, buff, sizeof(buff));
			//line of code that will allow us to open our files  on mac devices, using the undeclared mutex
			FD.open(Name, ios::in);
			//we put waiting the possible process
			sem_wait(&mutex);
			file_send_id++;
			int tid=file_send_id;
			sem_post(&mutex);
			
			cout<<"sending file. Transfer id = "<<tid<<endl;
			if(FD){
				if(!FD.is_open())
				{
					strcpy(buff, "0");
					write(sockfd, buff, sizeof(buff));
					cout<<"\nCould not open file"<<endl;
				}
				else
				{
					strcpy(buff, "1");
					write(sockfd, buff, sizeof(buff));
					
					string checksum = getMd5sum(Name);
					// send checksum of file to reciever
					strcpy(buff,checksum.c_str());
					write(sockfd, buff, sizeof(buff)); 
				
					int flag = 0;
					FD.seekg(0, ios::end);
	   				int file_size = FD.tellg();
	   				//cout<<"FILE SIZE "<<file_size<<endl;
	   				strcpy(buff,to_string(file_size).c_str());
	   				write(sockfd, buff, sizeof(buff)); 
	   				char file_contents[file_size]="";
	   				FD.close();
	   				FD.open(Name, ios::in);
					while(!FD.eof()) {
						
						sem_wait(&mutex);
						if(quit==1 || abort_id==tid)
						{
							sem_post(&mutex);
							close(sockfd);
							pthread_exit(NULL);
						}
						if(pause_id==tid && resume_id!=tid)
						{
							flag=0;
						}
						else
							flag=1;
						sem_post(&mutex);
						
						if(flag==1)
						{
							getline(FD, fromFile);
							//cout<<"File contents "<<fromFile<<endl;
							bzero(buff, sizeof(buff));
							n = 0; 
							// copy server message in the buffer 
							while ((fromFile[n]) != '\0') {
								buff[n] = fromFile[n]; 
								n++;
							}
							
							//strcat(file_contents,"\n");
							write(sockfd, buff, sizeof(buff)); 
							
							//read(sockfd, buff, sizeof(buff)); 
							strcat(buff,"\n");
							strcat(file_contents,buff);
							fromFile.clear();
						}
					}
					cout<<"File sent successfully. Transfer id: "<<tid<<endl;
					cout<<"File contents were : "<<endl;
					cout<<file_contents<<endl;
					
				}
			}
			else{
				strcpy(buff, "1");
				write(sockfd, buff, sizeof(buff));
				cout << "Please enter valid file name" << endl;
			}
	close(sockfd);
	sem_wait(&mutex);
	thread_count--;
	sem_post(&mutex);
	pthread_exit(NULL);

}

void * recv_file(void * param)
{

	string argv2 = *(string*)param;
	
	int sockfd, connfd; 
	struct sockaddr_in servaddr, cli; 

	// socket create and varification 
	sockfd = socket(AF_INET, SOCK_STREAM, 0); 
	if (sockfd == -1) { 
		printf("socket creation failed...\n"); 
		exit(0); 
	} 
	else
		printf("Socket successfully created..\n"); 
	bzero(&servaddr, sizeof(servaddr)); 

	// assign IP, PORT 
	servaddr.sin_family = AF_INET; 
	servaddr.sin_addr.s_addr = inet_addr("127.0.0.1"); 
	servaddr.sin_port = htons(PORT); 

	// connect the client socket to server socket 
	if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr)) != 0) { 
		printf("connection with the server failed...\n"); 
		exit(0); 
	} 
		
			char buff[MAX]; 
			int n; 
			fstream FD;
			string fromFile, Name, toFile, choice;
			strcpy(buff,"recv\n");
			write(sockfd, buff, sizeof(buff)); 
			
			bzero(buff, sizeof(buff)); 
			printf("Enter File name : "); 
			n = 0; 
			Name=argv2;
			strcpy(buff,Name.c_str());
			write(sockfd, buff, sizeof(buff)); 
			Name = buff;
			
			//check the file name what we send
			cout << "File name is : " << Name;
			
			
			
			//Receive acknowledge from server that it receive or not
			read(sockfd, buff, sizeof(buff)); 
			string check = buff;
			int flag=0;
			if(check == "1"){
				cout << "File found" << endl;
				
				read(sockfd, buff, sizeof(buff)); //read checksum
				string checksum = buff;
				
				string path="./hot_list_recv/";
		
				FD.open(path+Name, ios::out );
				
				read(sockfd, buff, sizeof(buff)); //read the size
				int file_size = atoi(buff);
				int size=0;
				//cout<<"file sizr "<<file_size<<endl;
				char file_contents[file_size]="";
				bzero(buff, sizeof(buff));
				while(size < file_size){
				
					
					
					sem_wait(&mutex);
					if(quit == 1)
					{
						thread_count--;
						sem_post(&mutex);
						pthread_exit(NULL);
					}
					sem_post(&mutex);
					
					//Receive content from server
					read(sockfd, buff, sizeof(buff)); 
					size = size + strlen(buff)+1;
					strcat(buff, "\n");
					toFile = buff;
					FD << toFile;
					
					strcat(file_contents,buff);	
					
					if(strlen(buff)==0){
						
						break;
					}
					bzero(buff, sizeof(buff));
					toFile.clear();
								
				}
				if(size != file_size)
				{
					FD.close();
					remove((path+Name).c_str());
				}
				cout<<"File successfully recieved"<<endl;
				cout<<"File contents are :"<<endl;
				cout<<file_contents<<endl;
				FD.close();
				string new_checksum=getMd5sum(path+Name);
				
				if(checksum == new_checksum)
				{
					cout<<"File received correctly"<<endl;
				}
				else
					cout<<"File received with errors"<<endl;
				
				
			}
			else if(check == "0"){
				cout << "\nFile NOT found\n";
			}
			
	close(sockfd);
	sem_wait(&mutex);
	thread_count--;
	sem_post(&mutex);
	pthread_exit(NULL);
		
}


void * get_file_list(void * param)
{

	
	
	int sockfd, connfd; 
	struct sockaddr_in servaddr, cli; 

	// socket create and varification 
	sockfd = socket(AF_INET, SOCK_STREAM, 0); 
	if (sockfd == -1) { 
		printf("socket creation failed...\n"); 
		exit(0); 
	} 
	else
		printf("Socket successfully created..\n"); 
	bzero(&servaddr, sizeof(servaddr)); 

	// assign IP, PORT 
	servaddr.sin_family = AF_INET; 
	servaddr.sin_addr.s_addr = inet_addr("127.0.0.1"); 
	servaddr.sin_port = htons(PORT); 

	// connect the client socket to server socket 
	if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr)) != 0) { 
		printf("connection with the server failed...\n"); 
		exit(0); 
	} 
		
			char buff[MAX]; 
			int n; 
			fstream FD;
			string fromFile, Name, toFile, choice;
			strcpy(buff,"recv\n");
			write(sockfd, buff, sizeof(buff)); 
			
			bzero(buff, sizeof(buff)); 
			printf("Enter File name : "); 
			n = 0; 
			cin>>Name;
			strcpy(buff,Name.c_str());
			write(sockfd, buff, sizeof(buff)); 
			Name = buff;
			
			//check the file name what we send
			cout << "File name is : " << Name;
			
			
			
			//Receive acknowledge from server that it receive or not
			read(sockfd, buff, sizeof(buff)); 
			string check = buff;
			int flag=0;
			if(check == "1"){
				cout << "File found" << endl;
				
				read(sockfd, buff, sizeof(buff)); //read checksum
				string checksum = buff;
				
				string path="./client_receive/";
		
				FD.open(path+Name, ios::out );
				
				read(sockfd, buff, sizeof(buff)); //read the size
				int file_size = atoi(buff);
				int size=0;
				//cout<<"file sizr "<<file_size<<endl;
				char file_contents[file_size]="";
				bzero(buff, sizeof(buff));
				while(size < file_size){
				
					
					
					sem_wait(&mutex);
					if(quit == 1)
					{
						thread_count--;
						sem_post(&mutex);
						pthread_exit(NULL);
					}
					sem_post(&mutex);
					
					//Receive content from server
					read(sockfd, buff, sizeof(buff)); 
					size = size + strlen(buff)+1;
					strcat(buff, "\n");
					toFile = buff;
					FD << toFile;
					
					strcat(file_contents,buff);	
					
					if(strlen(buff)==0){
						
						break;
					}
					bzero(buff, sizeof(buff));
					toFile.clear();
								
				}
				if(size != file_size)
				{
					FD.close();
					remove((path+Name).c_str());
				}
				cout<<"File successfully recieved"<<endl;
				cout<<"File contents are :"<<endl;
				cout<<file_contents<<endl;
				FD.close();
				string new_checksum=getMd5sum(path+Name);
				
				if(checksum == new_checksum)
				{
					cout<<"File received correctly"<<endl;
				}
				else
					cout<<"File received with errors"<<endl;
				
				
			}
			else if(check == "0"){
				cout << "\nFile NOT found\n";
			}
			
	close(sockfd);
	sem_wait(&mutex);
	thread_count--;
	sem_post(&mutex);
	pthread_exit(NULL);
		
}

void * func(void* arg) 
{ 
	struct parameter *p =(parameter*)arg;
	string argv1 =p->argv1 ;
	string argv2 =p->argv2 ;
	cout << "进来了" << argv1 <<"||" <<argv2<< endl ;
	char buff[MAX]; 
	int n; 
	fstream FD;
	string fromFile, Name, toFile, choice;

		bzero(buff, sizeof(buff)); 
		



	if (argv1=="upload"){


				//write(sockfd, buff, sizeof(buff)); 
				pthread_t thread1;
				
				sem_wait(&mutex);
				thread_count++;
				sem_post(&mutex);
				int ret1 = pthread_create(&thread1, NULL, send_file, &argv2);
				if (ret1!=0)
				{
					printf("Error In Creating Thread with ID:%d\n",1);
				}
				//pthread_join(thread1,NULL);
				
				sleep(2);

	}else if (argv1=="download"){
				pthread_t thread1;
				
				sem_wait(&mutex);
				thread_count++;
				sem_post(&mutex);
				int ret1 = pthread_create(&thread1, NULL, recv_file, &argv2);
				if (ret1!=0)
				{
					printf("Error In Creating Thread with ID:%d\n",1);
				}
				//pthread_join(thread1,NULL);
				
				sleep(2);

	}else{

		cout <<"show" <<endl;

	}


		
	
		
	
	pthread_exit(NULL);
} 



int main(int argc, char ** argv) 
{ 

	// if (argc!=2){

	// 		std::cout << "Wrong input" << std::endl;
	// 		return -1;

	// }
	// char * var1=argv[1];
	// char * var2=argv[2];
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	string var1=FLAGS_type;
	string var2=FLAGS_file;
	string port=FLAGS_port;
	struct parameter *par =  new parameter;
	par->argv1 = var1;

	if (par->argv1 != "get_file_list"){
		par->argv2 = var2;
		string s =par->argv2.c_str();
		char * k = NULL;
		
		int i_1;
		int i ;
		int stat_hot;
		if (s[s.size() - 2] - '0' >=0 && s[s.size() - 2] - '0' <=9 ){
			stat_hot = (s[s.size() - 5] =='h' && s[s.size() - 4] =='o' &&s[s.size() - 3] =='t' );
			 
			
		}
		else{

			 stat_hot = (s[s.size() - 4] =='h' && s[s.size() - 3] =='o' &&s[s.size() - 2] =='t' );
		}
		i_1=s[s.size() - 2] - '0'  ;
		i=s[s.size() - 1] - '0' -1 ;
		if (i_1<=9 && i_1>=0){
			i=i_1*10+i;

		}
		else{
			i=s[s.size() - 1] - '0' -1 ;
		}
		i=i+1; //后缀大小

		

		cout << stat_hot << "stat_hot"<< endl;
		if ( !(i<=99 && i>=1 || !stat_hot ) || !(  s[3] - '0' >=1 &&  s[3] - '0'<=9 ) ||  !(  s[s.size() - 2] - '0' >=1 &&  s[s.size() - 2] - '0'<=9 ) ||  !(  s[s.size() - 1] - '0' >=1 &&  s[s.size() - 1] - '0'<=9 ) ){
			cout << "错误文件命令" << endl;
			return 0 ;
		}
	}
	else{
		par->argv2 = "none";
	}
	

		

	if (var1=="get_file_list"){
		cout << "exiting file_list is" << endl;

	int sockfd, connfd; 
	sem_init(&mutex,0,1);
	struct sockaddr_in servaddr, cli; 
	pthread_t thread;
	
	sem_wait(&mutex);
	thread_count++;
	sem_post(&mutex);
	
	// function for chat 
	int ret1 = pthread_create(&thread, NULL, func, (void*)par);
	if (ret1!=0)
	{
		printf("Error In Creating Thread with ID:%d\n",1);
	}
	
	pthread_join(thread,NULL);
	sleep(2);
	return 0;


	}
	else if(var1=="upload"){
		cout << "upload file " << var2 << endl;
	int sockfd, connfd; 
	sem_init(&mutex,0,1);
	struct sockaddr_in servaddr, cli; 
	pthread_t thread;
	
	sem_wait(&mutex);
	thread_count++;
	sem_post(&mutex);
	
	// function for chat 
	int ret1 = pthread_create(&thread, NULL, func, (void*)par);
	if (ret1!=0)
	{
		printf("Error In Creating Thread with ID:%d\n",1);
	}
	
	pthread_join(thread,NULL);
	sleep(2);
	return 0;



	}
	else if(var1=="download"){
		cout << "down file " << var2 << endl;
	int sockfd, connfd; 
	sem_init(&mutex,0,1);
	struct sockaddr_in servaddr, cli; 
	pthread_t thread;
	
	sem_wait(&mutex);
	thread_count++;
	sem_post(&mutex);
	
	// function for chat 
	int ret1 = pthread_create(&thread, NULL, func, (void*)par);
	if (ret1!=0)
	{
		printf("Error In Creating Thread with ID:%d\n",1);
	}
	
	pthread_join(thread,NULL);
	sleep(2);
	return 0;


	}
	else{
		cout << "error input " << var2 << endl;
		return -1;
	}




} 

