#pragma once
#include <vector>
#include <unordered_map>
using namespace std;

namespace victor {

    class Node
    {
    public:
        int w;   //the symbol at this node

        int idx; //the index of this node

        Node* fater_node, * fail_node;    //Pointer to father node and fail node
        double fail_cost;                  //The cost when fail
        //Please note that the fail_cost and fail_node will be changed by re_route_failpath function using Aho - Corasick algorithm during Step2

        vector<Node*> next_nodes;        //All the possible next node from this node
        vector<double> next_nodes_cost;
        bool next_node(int w, Node** next_node, double* cost);
        // return wether w is in the next node of current node. 
        //if true change the two pointer to point at the next node ant its cost
        //if not change the two pointer to point at the fail node ant fail cost

        unordered_map<int, pair<double, int>> outputw_2_scrore_next_states;   //w --> <cost,next node's idx>
        //The map from symbol to its corresponding cost and next node's index at this node
        //This is generated at Step3

    };



    struct Score_output
    {

        unordered_map<int, pair<double, int>>* outputw_2_scrore_next_states;
        double return_cost;     //the score of the rest of the symbols.
    };


    class HWScorer
    {

    public:
        
        Node root_node;
        HWScorer(vector<vector<int>> hotwords,double weight);
        ~HWScorer();    //Destructor
        int init_state(void);
        Score_output score(int state_int);

        vector<Node*> all_nodes;     //All the nodes in the Tire tree 

    private:
        void add_hotword(vector<int> hotword);//adding one hotword to the tire tree
        double gen_score(int i);//generate the award at i th symbol.
        double weight_ = 1;   //This is the weight of hotword score
        void BFS_re_route_failpath(void);  //go through the Tire tree with BFS algorithm and reroute all the fail path using Aho-Corasick algorithm
        void re_route_failpath(Node* node);  //reroute the fail path of this node using Aho - Corasick algorithm
        void optimize_node(Node* node); //optimize this node 

    };



}