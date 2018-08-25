/**
Subsampling for manifold algorithms for word embeddings
**/
#include<iostream>
#include<string>
#include<fstream>
#include <sstream>
#include <unordered_map>
#include <math.h>
using namespace std;

int main() {


    float sample=1e-3;
    string line;
    string raw_filename="clean.txt";
    string raw_vocabname="vocabNYT.txt";
    ifstream inf(raw_vocabname);
    ofstream outf("clean-sub"+to_string(sample)+".txt");

    unordered_map<string, double> freq={};
    unordered_map<int, string> inv_vocab={};

    unsigned long long next_random = (long long) 35613983015;

    getline(inf, line);
	string word;
    istringstream iss(line);
    int count=0;

	do
    { 
        string word;
        iss >> word;
        if (!word.empty()) {
        	//cout << "Substring: \'" << word <<"\'"<< endl;
        	freq[word]=0.0;
            inv_vocab[count++]=word;
        }


     } while (iss);
    //cout<<count<<"  word"<<endl;

    getline(inf, line);
    stringstream ss(line);
    //cout<<line<<endl;
    count=0;
	while(ss)
    { 
        double frequency=-1;
        ss >> frequency;
        if (frequency<0) break;
        //cout<< frequency<<"count "<< count<< " "<<inv_vocab[count]<<endl;
        //cout << "Substring: \'" << frequency <<"\'"<< endl;
        freq[inv_vocab[count++]]=frequency;
    }
    /*cout<<freq.size()<<" freq size"<<endl;
    cout<<inv_vocab.size()<<" vocab size"<<endl;
    cout<<inv_vocab[37360]<<" 37360"<<endl;
    cout<<freq["anarchism"]<<endl;
    cout<<"Done"<<endl;*/
    inf.close();

    inf.open(raw_filename);
    
    count=0;
    iss.clear();
    while ( getline ( inf, line )) {
        //cout<<line<<endl;
        iss.str(line);
        //cout<< endl<<iss.str()<<endl << "New: ";
        do
        { 
            string word;
            iss >> word;
            float ran = (sqrt(freq[word]/sample) + 1) *sample/freq[word];
            next_random = next_random * (unsigned long long)25214903917 + 11;
            if ((!word.empty())) {
            //if ((ran < (next_random & 0xFFFF) / (float)65536)) cout<<ran <<" :"<< (next_random & 0xFFFF) / (float)65536<<" "<< word<<endl;
               if (freq.end()==freq.find(word)||(! (ran < (next_random & 0xFFFF) / (float)65536)) )outf << " " << word;
            }

        } while (iss);
        iss.clear();
        outf<<endl;
        //if (count++>2) break;
        }
    cout<<freq["the"]<<endl;
    inf.close();
    outf.close();

}
