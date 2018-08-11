/**
Subsampling for manifold algorithms for word embeddings
**/
#include<iostream>
#include<string>
#include<fstream>
#include <sstream>
#include <unordered_map>
using namespace std;

int main() {

    string line;
    ifstream inf("vocab9.txt");
    ofstream outf("enwik9-sub.txt");

    unordered_map<string, double> freq={};
    unordered_map<int, string> inv_vocab={};

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

    inf.open("enwik9.txt");
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
            if (!word.empty()) {
                outf << " " << word;
            }

        } while (iss);
        iss.clear();
        outf<<endl;
        //if (count++>2) break;
        }
    inf.close();
    outf.close();

}
