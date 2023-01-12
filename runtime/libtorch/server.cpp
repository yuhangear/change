
// g++ client.cpp -lpthread -o server

// ./server


#include <stdio.h> 
#include <netdb.h>
#include <netinet/in.h> 
#include <stdlib.h> 
#include <string.h> 
#include <sys/socket.h> 
#include <sys/types.h> 
#include <unistd.h>
#include <iostream>
#include <pthread.h>
#include <fstream>
#include <string>
#include <semaphore.h>
#define MAX 1024
#define PORT 5001	
#define SA struct sockaddr 

using namespace std;
pthread_mutex_t mutex;

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

void * send_file(void * client_sock)
{
	fstream FD;
	
	int sockfd = * ((int * ) client_sock);
	string convert_msg = "", fileName = "", toFile = "", fromFile = "", command, Name;
	char msg[500] = "", buff[1024] = "", sendTo[1024] = "";
	char msg1[500] = "You have typed type send commnd";
	char msg2[500] = "You have typed recv commnd";	
	int len;
	
	
	read(sockfd, buff, sizeof(buff)); 
	// print buffer which contains the client contents 
	printf("File name is: %s", buff); 
	Name = buff;
	cout << "\nName of file is :" << Name << endl;
	
	
	
	read(sockfd, buff, sizeof(buff)); //read checkfield
	string check = buff;
	cout << "Check field is " << check<<endl;
	
	
	if(check == "1"){
		read(sockfd, buff, sizeof(buff)); //read checksum
		string checksum = buff;
		
		string path="./hot_word_dir_old/";
		
		FD.open(path+Name, ios::out );
		int size =0;
		
		read(sockfd, buff, sizeof(buff)); 
		int file_size = atoi(buff);
		//cout<<"file size : " << file_size << endl;
		char file_contents[file_size]="";
		while(size < file_size){
		
			memset(&buff, sizeof(buff), 0);
			
			read(sockfd, buff, sizeof(buff)); 
			size = size + strlen(buff)+1;
			
			strcat(buff, "\n");
			toFile = buff;
			FD << toFile;
			
			strcat(file_contents,buff);
			
			bzero(buff, sizeof(buff));
			toFile.clear();
			
		}
		if(size != file_size)
		{
			FD.close();
			remove((path+Name).c_str());
		}
		
		cout<<"File successfully recieved\n"<<endl;
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
	
	pthread_exit(NULL);
}

void * recv_file(void * client_sock)
{

	fstream FD;
	
	int sockfd = * ((int * ) client_sock);
	string convert_msg = "", fileName = "", toFile = "", fromFile = "", command, Name;
	char msg[500] = "", buff[1024] = "", sendTo[1024] = "";
	char msg1[500] = "You have typed send commnd";
	char msg2[500] = "You have typed recv commnd";	
	int len;
	
	
	read(sockfd, buff, sizeof(buff)); 
	// print buffer which contains the client contents 
	printf("File name is: %s", buff); 
	cout << endl;
	Name = buff;
	cout << "\nName of file is :" << Name << endl;
	FD.open(Name, ios::in);
		if(FD){
			strcpy(buff, "1");
			write(sockfd, buff, sizeof(buff));
			
			//md5sm functions
			string checksum = getMd5sum(Name);
			// send checksum of file to reciever
			strcpy(buff,checksum.c_str());
			write(sockfd, buff, sizeof(buff)); 
			
			FD.seekg(0, ios::end);
			int file_size = FD.tellg();
			//cout<<"FILE SIZE "<<file_size<<endl;
			strcpy(buff,to_string(file_size).c_str());
			write(sockfd, buff, sizeof(buff)); 
			FD.close();
			FD.open(Name, ios::in);
			char file_contents[file_size]="";
			while(!FD.eof()){
				getline(FD, fromFile);
				
				bzero(buff, sizeof(buff));
				int n = 0; 
				// copy server message in the buffer 
				while ((fromFile[n]) != '\0') {
					buff[n] = fromFile[n]; 
					n++;
				}
				
				write(sockfd, buff, sizeof(buff)); 
				//cout << "Question send to client :" << buff << '\n';
				strcat(buff,"\n");
				strcat(file_contents,buff);
				fromFile.clear();
			}
			//Prints the contents of the buffer on console
			cout<<"File contents were : "<<endl;
			cout<<file_contents<<endl;
			
		}
		else{
			strcpy(buff, "0");
			write(sockfd, buff, sizeof(buff));
			cout << "This file not found in server side";
		}
		//once files are sent or when the excecutable detects that the file is not found on the server side, we close existing thread
		pthread_exit(NULL);
}


// Driver function 
int main() 
{ 
	int sockfd, connfd, len; 
	struct sockaddr_in servaddr, cli; 
	pthread_t recvt;

	char buff[MAX]; 
	string command;
	// socket create and verification 
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
	servaddr.sin_addr.s_addr = htonl(INADDR_ANY); 
	servaddr.sin_port = htons(PORT); 

	// Binding newly created socket to given IP and verification 
	if ((bind(sockfd, (SA*)&servaddr, sizeof(servaddr))) != 0) { 
		printf("socket bind failed...\n"); 
		exit(0); 
	} 
	else
		printf("Socket successfully binded..\n"); 

	// Now server is ready to listen and verification 
	if ((listen(sockfd, 5)) != 0) { 
		printf("Listen failed...\n"); 
		exit(0); 
	} 
	else
		printf("Server listening..\n"); 
	len = sizeof(cli); 

	while(1){
		printf("Server listening..\n"); 
		bzero(buff, sizeof(buff));
		// Accept the data packet from client and verification 
		connfd = accept(sockfd, (struct sockaddr * ) NULL, NULL);
		if (connfd < 0) { 
			printf("server acccept failed...\n"); 
			exit(0); 
		} 
		else
		    //if it gets accpeted... we hold on with the process
			printf("server acccept the client...\n"); 

		//cout<<"SOCKET "<<sockfd<<endl;
		// read the message from client and copy it in buffer 
		read(connfd, buff, sizeof(buff)); 
		// print buffer which contains the client contents 
		printf("From client choice: %s", buff); 
		command = buff;
		
		if(command == "send\n"){
			pthread_create( & recvt, NULL, &send_file, & connfd);
		}
		else if(command == "recv\n"){
			pthread_create( & recvt, NULL, &recv_file, & connfd);
		}
		//pthread_create( & recvt, NULL, &func, & connfd);
		pthread_mutex_unlock( & mutex);
		
	}

	// After chatting, close the socket 
	close(sockfd); 
} 

