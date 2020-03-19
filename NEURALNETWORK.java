import java.util.Arrays;
import java.util.Collections;

public class NEURALNETWORK {

	static int inputN = 25;
	static int hiddenN = 35;
	static int outputN = 3;
	static enum neuronType {In, Hid, Out};
	
	static double n = 0.5;
	static double lambda = 1;
	
	public NEURON[] neurons = new NEURON[inputN + hiddenN + outputN];
	
	public NEURALNETWORK() {
		//Input neurons
		for (int i = 0; i < inputN; i++) {
			neurons[i] = new NEURON(neuronType.In);
		}
		//Hidden neurons
		for (int i = inputN; i < inputN + hiddenN; i++) {
			neurons[i] = new NEURON(neuronType.Hid);
		}
		//Output neurons
		for (int i = inputN + hiddenN; i < inputN + hiddenN + outputN; i++) {
			neurons[i] = new NEURON(neuronType.Out);
		}
	}
	
	public double[] getFinalOuts() {
		double[] result = new double[outputN];
		for (int i = 0; i < outputN; i++) {
			result[i] = neurons[i].getOutput();
		}
		return result;
	}
	
	public NEURON[] getNeurons() {
		return neurons;
	}
	
	public void forwardProp(double[] inputSet) {
		for (int i = 0; i < neurons.length; i++) {
			double val = 0;
			double net = 0;
			double newOut = 0;
			if (neurons[i].type == neuronType.In) {
				// Setting output of input neurons to the corresponding input set values
				neurons[i].setOutput(inputSet[i]);
			}
			if (neurons[i].type == neuronType.Hid) {
				// net = sum of the output for input neuron j * weight to hidden neuron i from input neuron j
				// 		 for all connections leading into hidden neuron i
				// 25 total weight/output sumss for each hidden neuron, 35 weights per list
				// This if statement will be accessed 35 times
				// i begins at 25 and ends at 59, j will count from 0 to 34
				// net calculation
				for (int j = 0; j < inputN; j++) {
					val += (neurons[j].getOutput() * neurons[j].getWeights()[i - inputN]);
				}
				net = (neurons[i].getBias() * neurons[i].getBiasWeight()) + val;
				// Apply activation fucntion to net (weighted sum) of input connections
				newOut = 1.0 / (1.0 + Math.exp(-net * lambda));
				neurons[i].setOutput(newOut);
			}
			if (neurons[i].type == neuronType.Out) {
				// net = sum of the output for hidden neuron j * weight to output neuron i from hidden neuron j
				//		 for all connections leading into output neuron i
				// 35 total weight/output sums for each hidden neuron, 3 weights per list
				// This if statement will be accessed 3 times
				// i begins at 59 and ends at 62, j will count from 25 to 59
				// net calculation
				for (int j = inputN; j < inputN + hiddenN; j++) {
					val += (neurons[j].getOutput() * neurons[j].getWeights()[i - inputN - hiddenN]);
				}
				net = (neurons[i].getBias() * neurons[i].getBiasWeight()) + val;
				// Apply activation function to net (weighted sum) of input connections
				newOut = 1.0 / (1.0 + Math.exp(-net * lambda));
				neurons[i].setOutput(newOut);
			}
		}		
	}
	
	public void backProp(double[] targetResult) {
		// Calculates error for output neurons in reverse order first
		for (int i = neurons.length - 1; i > inputN + hiddenN - 1; i--) {
			double newError = 0;
			double delWeight = 0;
			double newWeight = 0;
			if (neurons[i].type == neuronType.Out) {
				// Calculate error for output neurons compared to target output, weight adjustments
//				double loss = Math.abs(targetResult[i-inputN-hiddenN] - neurons[i].getOutput());
//				System.out.println(loss);
				newError = ((targetResult[i - inputN - hiddenN] - neurons[i].getOutput())
							* neurons[i].getOutput() *
							(1 - neurons[i].getOutput()));
				neurons[i].setError(newError);
				//System.out.println(neurons[i].getError());
				
				// Calculate and apply weight adjustments for hidden neuron to out neuron weights
				// delWeight = learning rate * output neuron ERROR * hidden neuron OUTPUT
				for (int j = inputN + hiddenN - 1; j > inputN - 1; j--) {
					delWeight = n * neurons[i].getError() * neurons[j].getOutput();
					newWeight = neurons[j].getWeights()[i - inputN - hiddenN] + delWeight;
					neurons[j].setWeight(i - inputN - hiddenN, newWeight);
				}
				// Bias weight adjustments
//				delWeight = n * neurons[i].getOutput() * neurons[i].getBias();
//				newWeight = neurons[i].getBiasWeight() + delWeight;
//				neurons[i].setBiasWeight(newWeight);
			}
			else {
				System.out.println("Error in backProp(): Not output neuron");
			}
		}
		// Calculates error for hidden neurons, weight adjustments
		for (int i = inputN; i < inputN + hiddenN; i++) {
			double newError = 0;
			double delWeight = 0;
			double newWeight = 0;
			double sum = 0;
			if (neurons[i].type == neuronType.Hid) {
				// Hidden neuron sum = sum of error for corresponding output neuron * corresponding weight value
				for (int j = 0; j < outputN; j++) {
					sum += (neurons[j + inputN + hiddenN].getError() * neurons[i].getWeights()[j]); 
				}
				newError = ((neurons[i].getOutput() * (1 - neurons[i].getOutput()) * sum));
				neurons[i].setError(newError);
				//System.out.println(neurons[i].getError());
				
				// Calculate and apply weight adjustments for input neuron to hidden neuron weights
				// delWeight = learning rate * hidden neuron ERROR * input neuron OUTPUT				
				for (int j = 0; j < inputN; j++) {
					delWeight = n * neurons[i].getError() * neurons[j].getOutput();
					newWeight = neurons[j].getWeights()[i - inputN] + delWeight;
					neurons[j].setWeight(i - inputN, newWeight);
				}
				// Bias weight adjustments
//				delWeight = n * neurons[i].getOutput() * neurons[i].getBias();
//				newWeight = neurons[i].getBiasWeight() + delWeight;
//				neurons[i].setBiasWeight(newWeight);
			}
			else {
				System.out.println("Error in backProp(): Not hidden neuron");
			}
		}
	}

	public int classify(double[] result) {
		double[] diffList = {0,0,0,0,0,0};
		for (int i = 0; i < DRIVER.testsC.length; i++) {
			double diff = 0;
			for (int j = 0; j < outputN; j++) {
				diff += Math.abs(DRIVER.testsC[i][1][j] - result[j]);
			}
			diffList[i] = diff;
		}
		//System.out.println(Arrays.toString(diffList));
		double min = getMin(diffList);
		for (int i = 0; i < diffList.length; i++) {
			if (diffList[i] == min) { 
				return i;
			}
		}
		return -1;
	}
	
	public double getMin(double[] diffList) {
		double min = diffList[0];
		for (int i = 1; i < diffList.length; i++) {
			if (diffList[i] < min) {
				min = diffList[i];
			}
		}
		return min;
	}
	
	public double getTSSE(double[] result) {
		double sumOut = 0;
		double sumPat = 0;
		for (int i = 0; i < DRIVER.testsC.length; i++) {
			for (int j = 0; j < DRIVER.testsC[i][1].length; j++) {
				sumOut += Math.pow(DRIVER.testsC[i][1][j] - result[j], 2);
			}
			sumPat += sumOut;
		}
		return 0.5*sumPat;
	}
	
	public double getRMSE(double TSSE) {
		double RMSE = Math.sqrt((TSSE * 2)/(6 * 3));
		return RMSE;
	}
}
