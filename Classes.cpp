#include<stdlib.h>
#include<stdio.h>
#include<vector>
#include<math.h>
#include <algorithm> 
#include "Classes.h"

/// activation functions
double sigmoid(double x) { return 1/(1+exp(-x)); }
double relu(double x) { if (x > 0) return x; else return 0;}
double tanh(double x){return (2 / (1 + exp(-2*x))) - 1;}

double sigmoid_prime(double x){return sigmoid(x)*(1-sigmoid(x));}

/// NEURON FUNCTIONS
Neuron::Neuron(){
    activation = 0;
    row = NULL;
    col = NULL;
    parent = NULL;
}
Neuron::Neuron(double a, int c, int r, Layer*p){
    activation = a;
    row = r;
    col = c;
    parent = p;
}
double Neuron::getActivation(){
    return activation;
}
void Neuron::setActivation(double a){
    activation = a;
}
int Neuron::getCol(){
    return col;
}
int Neuron::getRow(){
    return row;
}
void Neuron::setCol(int a){
    col = a;
}
void Neuron::setRow(int a){
    row = a;
}
Layer* Neuron::getParent(){
    return parent;
}
void Neuron::setParent(Layer* p){
    parent = p;
}
void Neuron::printNeuron(){
    printf("[(C, R) = (%d, %d), A = %f] ", col, row, activation);
    printf("Weights:[");
    for (double i: weights){
       printf("%f,", i);
    }
    printf("]\n");
        
}

void Neuron::setWeights(vector<double> w){
    // if(parent->getPrev()==NULL){
    //     printf("no layer before this one\n");
    //     return;
    // }
    weights = w;
    // if (w.size() == parent->getPrev()->getNeurons().size()){
    //     // weights.clear();
    //     // for(int i = 0; i<w.size(); i++){
    //     //     weights.push_back(w.at(i));
    //     // }
    // }
    // else{
    //     printf("size of weight vector (%d) does not match number of neurons in previous layer(%d)\n", w.size(),parent->getPrev()->getNeurons().size());
    // }
}
void Neuron::setWeights(int a, int r){
    if(parent->getPrev()==NULL){
        printf("no layer before this one\n");
        return;
    }
    if(weights.size()!=parent->getPrev()->getNeurons().size()){
        printf("First initialize weights array!\n");
        printf("Size of weights vector: %d, size of neuron vector: %d\n", weights.size(), parent->getPrev()->getNeurons().size());
    }
    else{
        weights.at(r)= a;
    }
}
vector<double> Neuron::getWeights(){
    return weights;
}
double Neuron::getWeightsIndex(int r){
    if (r>=weights.size()){
        printf("Index out of range, try again!; Index: %d, Size of Weight Vector: %d", r, weights.size());
        return -99;
    }
    else{
        return weights.at(r);
    }
}
/// LIST FUNCTIONS

List::List(){
    head = NULL;
    tail = NULL;
    activationType = 0;
    biases = {};
}
List::List(Layer*h){
    head = h;
    tail = h;
    activationType = 0;
    biases = {};
}
List::List(Layer*h, Layer*l){
    head = h;
    tail = l;
    activationType = 0;
    biases = {};
}
void List::setHead(Layer *n){
    head = n;
}
void List::setTail(Layer *n){
    tail = n;
}
Layer* List::getHead(){
    return head;
}
Layer* List::getTail(){
    return tail;
}
vector<double> List::getBiases(){
    return biases;
}

int List::getNumLayers(){
    int number = 0;
    if(head == nullptr ||tail == nullptr){
        return 0;
    }
    if (head == tail){
        return 1;
    }
    while(head!=tail){
        number++;
    }
    return number;
}
void List::addLayerTail(Layer*n){
    double ranValue = (std::fmod(rand(),10))/10;
    if (tail == NULL && head == NULL){
        tail = n;
        head = n;
        n->setCol(0);
        biases.push_back(ranValue);
        return;
    }
    else{
        tail->setNext(n);
        n->setPrev(tail);
        tail=n;
        Layer* temp = head;
        int index = 0;
        while(temp!= NULL){
            index++;
            temp = temp->getNext();
        }
        n->setCol(index-1);
        biases.push_back(ranValue);
    }

}
void List::addLayerHead(Layer*n){
    double ranValue = (std::fmod(rand(),10))/10;
    if (head==NULL && tail == NULL){
        tail = n;
        head = n;
        n->setCol(0);
        biases.push_back(ranValue);
        return;
    }
    else if (head == NULL&&head->getNext() != NULL){
        head = n;
        n->setCol(0);
        biases.push_back(ranValue);
        return;
    }
    else{
        n->setNext(head);
        head->setPrev(n);
        head = n;
        n->setCol(0);
        biases.push_back(ranValue);
    }
}
void List::setActivation(int a){
    if (a<3){
        activationType = a;
    }
    else{
        activationType = 0;
    }
    
}
void List::forwardProp(vector<double> input){
    Layer* temp = head;
    if(input.size() != head->getNeurons().size()){
        printf("input size not valid, must equal number of input neurons\n");
        printf("Size of input vector: %d\n", input.size());
        printf("size of input layer: %d\n", head->getNeurons().size());
        return;
    }
    else{
        double largest_element = INT_MIN;
        for (vector<double>:: iterator it = input.begin(); it != input.end(); it++)
            {
                if(*it > largest_element)
                {
                largest_element = *it;
                }
            }
        for (vector<double>:: iterator it = input.begin(); it != input.end(); it++)
            {
                (*it)/=largest_element;
            }
        int index = 0;
        while(true){
            if(index == input.size()){
                break;
            }
            head->getNeurons().at(index)->setActivation(input.at(index));
            index++;
        }
        int layerNum = 0;
        while(temp->getNext()!=NULL){
            double sum = 0;
            index = 0;
            vector<Neuron*> n = temp->getNeurons();
            for(int i = 0; i<temp->getNext()->getNeurons().size(); i++){
                sum = 0;
                for(int j = 0; j<n.size(); j++){
                    //LOOK
                    switch(activationType){
                        case 0:
                            sum+= n.at(j)->getActivation()* temp->getNext()->getNeurons().at(i)->getWeights().at(j);  //multiply weights
                            break;
                        default://*  *; temp->getNext()->getNeurons().at(i)->getWeights().at(j) + biases.at(temp->getCol());
                            sum+= n.at(j)->getActivation()* temp->getNext()->getNeurons().at(i)->getWeights().at(j);
                            break;
                    }
                    
                }
                
                temp->getNext()->getNeurons().at(i)->setActivation(sum);
            }
            temp= temp->getNext();
            layerNum++;
        }
    }
}
void List::printList(){
    Layer* temp = head;
    while(temp!=NULL){
        temp->printLayer();
        temp = temp->getNext();
    }
}

vector<double> List::cost_derivative(double a){
    Layer* temp = head;
    while(temp->getNext()!=NULL){
        temp = temp->getNext();
    }
    vector<Neuron*> n = temp->getNeurons();
    vector<double> output;
    for(int i = 0; i<n.size(); i++){
        output.push_back(n.at(i)->getActivation() - a);
    }
    return output;
}
vector<double> List::backprop(){
    vector<vector<double>> dBias;
    vector<vector<double>> dWeight;
    Layer* temp = head;
    // while(temp!=NULL){
    //     dBias.push_back(biases);
    //     vector<Neuron*> n = temp->getNeurons();
    //     for(int i = 0; i<n.size(); i++){
    //         dWeight.push_back(n.at(i)->getWeights());
    //     }
    //     temp = temp->getNext();
    // }
    // vector<double> zs;
    // vector<double> activation;

    // for()
    // return dBias.at(0);
    while(temp!=NULL){
        temp = temp->getNext();
    }
    vector<double> delta = cost_derivative()* sigmoid_prime()
}


///Layer Functions

Layer::Layer(){
    col = 0;
    next = NULL;
    prev = NULL;
    neurons = {new Neuron()};
}
Layer::Layer(Layer*n, Layer* p){
    col = 0;
    next = n;
    prev = p;
    neurons = {new Neuron()};
}

int Layer::getCol(){
    return col;
}
Layer* Layer::getNext(){
    return next;
}
Layer* Layer::getPrev(){
    return prev;
}
void Layer::setNext(Layer*n){
    next = n;
}
void Layer::setPrev(Layer*p){
    prev = p;
}
vector<Neuron*> Layer::getNeurons(){
    return neurons;
}
void Layer::setNeurons(vector<Neuron*> n){//LOOK
    neurons.clear();
    neurons = n;
    vector<double> vectors;
    // neurons.at(i)->setWeights(std::fmod(rand(), 10), j);
    if(this->prev!=NULL){
        for(int i = 0; i<neurons.size(); i++){
            for(int j = 0; j<this->prev->getNeurons().size(); j++){
                vectors.push_back(std::fmod(rand(), 10));
            }
            neurons.at(i)->setWeights(vectors);
            neurons.at(i)->setParent(this);
        }
    }

}
void Layer::deleteNeuron(int r){ //LOOK
    neurons.erase(neurons.begin() + r);
    for(int i = 0; i<this->next->getNeurons().size(); i++){
        this->next->getNeurons().at(i)->getWeights().erase(this->getNeurons().at(i)->getWeights().begin() + r); //erases weight associated with this neuron from each weight vector of every neuron in the next layer. Stupid. 
    }

}
void Layer::addNeuron(Neuron *n){ //LOOK
    neurons.push_back(n);
    int vecSize = neurons.size();
    if (n->getRow() == NULL || n->getRow()!= vecSize){
        n->setRow(vecSize);
    }
    if(n->getParent() == NULL || n->getParent() != this){
        n->setParent(this);
    }
    if(n->getCol()== NULL||n->getCol() != this->getCol()){
        n->setCol(this->getCol());
    }
    if (this->next != NULL){
        for(int i = 0; i<this->next->getNeurons().size(); i++){
            this->next->getNeurons().at(i)->getWeights().insert(this->getNeurons().at(i)->getWeights().end(), std::fmod(rand(), 10)); //erases weight associated with this neuron from each weight vector of every neuron in the next layer. Stupid. 
        }
    }
    if(this->prev!=NULL){
        vector<double> vectors;
        for(int i = 0; i<neurons.size(); i++){
            for(int j = 0; j<this->prev->getNeurons().size(); j++){
                vectors.push_back(std::fmod(rand(), 10));
            }
            neurons.at(i)->setWeights(vectors);
        }
    }
}
void Layer::initWeights(){
    if(this->next == NULL){
    }
    if (this->next != NULL){
        
        for(int i = 0; i<this->next->getNeurons().size(); i++){
            vector<double> t = {};
            for(int j = 0; j<this->getNeurons().size(); j++){
                t.push_back((double)rand()/RAND_MAX*2.0-1.0);
                //erases weight associated with this neuron from each weight vector of every neuron in the next layer. Stupid. 
            }
            this->next->getNeurons().at(i)->setWeights(t);
            //erases weight associated with this neuron from each weight vector of every neuron in the next layer. Stupid. 
        }
    }
    
    // if(this->prev!=NULL){
    //     vector<double> vectors;
    //     for(int i = 0; i<neurons.size(); i++){
    //         for(int j = 0; j<this->prev->getNeurons().size(); j++){
    //             vectors.push_back(std::fmod(rand(), 10));
    //         }
    //         neurons.at(i)->setWeights(vectors);
    //     }
    // }
}
Neuron* Layer::getNeuronIndex(int r){
    return neurons.at(r);
}
void Layer::printLayer(){
    int i = 0;
    while(true){
        vector<Neuron*> n = this->getNeurons();
        if(n.size() == 0){
            printf("empty");
            return;
        }
        if((i >= n.size())){
            return;
        }
        n.at(i)->printNeuron();
        i++;
    }
}
void Layer::setCol(int a){
    col = a;
    for(int i = 0; i<neurons.size(); i++){
        neurons.at(i)->setCol(a);
    }
}
