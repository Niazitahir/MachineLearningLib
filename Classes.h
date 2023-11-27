#ifndef Classes_h
#define Classes_h
#include <vector>
using namespace std;
class Layer;

class Neuron{
    private:
        int col;
        int row;
        double activation;
        Layer* parent;
        vector<double> weights;
    public:
        Neuron(double a, int col, int row, Layer* p);
        Neuron();
        double getActivation();
        int getCol();
        int getRow();
        void setRow(int a);
        void setCol(int a);
        void setParent(Layer* p);
        Layer* getParent();
        void setActivation(double a);
        void printNeuron();
        void setWeights(vector<double> w);
        void setWeights(int a, int r);
        vector<double> getWeights();
        double getWeightsIndex(int r);
};
class Layer{
    private:
        int col;
        Layer *next;
        Layer *prev;
        vector<Neuron*> neurons;

    public:
        Layer();
        Layer(Layer* n, Layer*p);
        int getCol();
        Layer* getNext();
        Layer* getPrev();
        void setNext(Layer* n);
        void setPrev(Layer* p);
        vector<Neuron*> getNeurons();
        void setNeurons(vector<Neuron*> n);
        void deleteNeuron(int r);
        void addNeuron(Neuron* n);
        Neuron* getNeuronIndex(int r);
        void printLayer();
        void setCol(int a);
        void initWeights();


};
class List{
    private:
        struct Layer *head;
        struct Layer *tail;
        vector<double> biases;
        int activationType;
    public:
        List();
        List(Layer *h);
        List(Layer *h, Layer *l);
        void setHead(Layer*n);
        void setTail(Layer*n);
        void addLayerTail(Layer *n);
        void deleteLayer(Layer *n);
        Layer* getHead();
        Layer* getTail();
        vector<double> getBiases();
        int getNumLayers();
        void printList();
        void addLayerHead(Layer* h);
        void setActivation(int a);//0 stands for sigmoid, 1 stands for relu, 2 stands for tanh
        void forwardProp(vector<double> input);
        vector<double> backprop();
        vector<double> cost_derivative(double a);

        
}; 
#endif