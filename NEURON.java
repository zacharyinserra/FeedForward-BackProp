import java.util.Random;

public class NEURON {
	
	Random r = new Random();
	
	public NEURALNETWORK.neuronType type;
	private double bias = 1;
	private double biasWeight = r.nextDouble() * 2 - 1;
	private double output;
	private double error;
	private double[] weights;
	
	public NEURON(NEURALNETWORK.neuronType type) {
		this.type  = type;
		this.bias = bias;
		this.biasWeight = biasWeight;
		this.output = output;
		this.error = error;
		
		// Initialize weights
		if (type == NEURALNETWORK.neuronType.In) {
			this.weights = new double[NEURALNETWORK.hiddenN];
			for (int i = 0; i < NEURALNETWORK.hiddenN; i++) {
				this.weights[i] = r.nextDouble() * 2 - 1;
			}
		}
		if (type == NEURALNETWORK.neuronType.Hid) {
			this.weights = new double[NEURALNETWORK.outputN];
			for (int i = 0; i < NEURALNETWORK.outputN; i++) {
				this.weights[i] = r.nextDouble() * 2 - 1;
			}
		}
		if (type == NEURALNETWORK.neuronType.Out) {
			this.weights = new double[1];
			this.weights[0] = 0;
		}
	}

	public NEURALNETWORK.neuronType getNeuronType(){
		return this.type;
	}
	public double[] getWeights() {
		return this.weights;
	}
	public void setWeight(int i, double newWeight) {
		this.weights[i] = newWeight;
	}
	public void setOutput(double val) {
		this.output = val;
	}
	public double getOutput() {
		return this.output;
	}
	public double getBias() {
		return this.bias;
	}
	public double getBiasWeight() {
		return this.biasWeight;
	}
	public void setBiasWeight(double newWeight) {
		this.biasWeight = newWeight;
	}
	public void setError(double val) {
		this.error = val;
	}
	public double getError() {
		return this.error;
	}
}
