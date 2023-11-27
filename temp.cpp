#include "Classes.h"
#include<iostream>
#include<stdio.h>
#include<time.h>
//use g++ to compile
int main(){
    srand(time(NULL));
    List *newList = new List();
    int numLayers = 5;
    int numNeurons = 5;
    for(int i = 0; i<numLayers; i++){
        Layer* layer = new Layer();
        for (int j = 0; j<5; j++){
            Neuron* neuron = new Neuron(((double)rand()/RAND_MAX*2.0-1.0), i, j, layer);
            //printf("[(C, R) = (%d, %d), A = %d]\n", neuron->getCol(), neuron->getRow(), neuron->getActivation());
            layer->addNeuron(neuron); //layer already has one on init
            //create layer function to auto make neurons with set number of neurons
            
        }
        newList->addLayerTail(layer);
    }
    Layer* temp = newList->getHead();
    while(temp != NULL){
        temp->initWeights();
        temp = temp->getNext();
    }
    vector<double> inputs = {0,1,2,3,4,5};
    newList->forwardProp(inputs);
    newList->printList();
    double b = 3;
    vector<double> n = newList->cost_derivative(b);
    vector<double> y = newList->backprop();
    return 0;

}