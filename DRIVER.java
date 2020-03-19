import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Random;

public class DRIVER {
	
	static Random r = new Random();
	
	// Input patterns and desired outputs based on six output neurons
	// Clean binary representation of plus +
	static double plusC[][] = {{0, 0, 1, 0, 0,
								0, 0, 1, 0, 0,
								1, 1, 1, 1, 1,
								0, 0, 1, 0, 0,
								0, 0, 1, 0, 0}, 
								{1, 0, 0}};
	// Clean binary representation of minus -
	static double minusC[][] = {{0, 0, 0, 0, 0,
								 0, 0, 0, 0, 0,
								 1, 1, 1, 1, 1,
								 0, 0, 0, 0, 0,
								 0, 0, 0, 0, 0}, 
								{0, 1, 0}};
	// Clean binary representation of forward slash \
	static double fslashC[][] = {{1, 0, 0, 0, 0,
								  0, 1, 0, 0, 0,
								  0, 0, 1, 0, 0,
								  0, 0, 0, 1, 0,
								  0, 0, 0, 0, 1}, 
								 {0, 0, 1}};
	// Clean binary representation of back slash /
	static double bslashC[][] = {{0, 0, 0, 0, 1,
								  0, 0, 0, 1, 0,
								  0, 0, 1, 0, 0,
								  0, 1, 0, 0, 0,
								  1, 0, 0, 0, 0}, 
								 {1, 1, 0}};
	// Clean binary representation of X
	static double xC[][] = {{1, 0, 0, 0, 1,
		  					 0, 1, 0, 1, 0,
		  					 0, 0, 1, 0, 0,
		  					 0, 1, 0, 1, 0,
		  					 1, 0, 0, 0, 1}, 
							{0, 1, 1}};
	// Clean binary representation of |
	static double lineC[][] = {{0, 0, 1, 0, 0,
		  						0, 0, 1, 0, 0,
		  						0, 0, 1, 0, 0,
		  						0, 0, 1, 0, 0,
		  						0, 0, 1, 0, 0}, 
								{1, 0, 1}};
	
	// Dirty binary representations
	static double noise = 0.9;
	static double plusD[][] = plusC;
	static double minusD[][] = minusC;
	static double fslashD[][] = fslashC;
	static double bslashD[][] = bslashC;
	static double xD[][] = xC;
	static double lineD[][] = lineC;

	static double testsC[][][] = new double[][][] {plusC, minusC, fslashC, bslashC, xC, lineC};
	static double testsD[][][] = new double[][][] {plusD, minusD, fslashD, bslashD, xD, lineD};
	
	static int epochs = 1;
	
	public static void main(String[] args) throws IOException {
		
		NEURALNETWORK NN = new NEURALNETWORK();
		NEURALNETWORK NN2 = new NEURALNETWORK();
		
		// Generates dirty set
		for (int i = 0; i < testsD.length; i++) {
			for (int k = 0; k < testsD[i][0].length; k++) {
				double val = testsD[i][0][k];							
				if (val == 0.0) {
					testsD[i][0][k] = val + (r.nextDouble() * noise);
				}
				else if (val == 1.0) {
					testsD[i][0][k] = val - (r.nextDouble() * noise);
				}
			}
		}

		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		boolean flip = true;
		while (flip) {
			System.out.println("");
			String in = br.readLine();
			switch (in) {
			case "trainclean":

				System.out.println("Noise level: " + noise);
				//Training clean set
				double[] result = {0,0,0};
//				System.out.println("Epochs: " + epochs);
//				for (int e = 0; e < epochs; e++) {
				for (int i = 0; i < testsC.length; i++) {
					System.out.println("Clean results:");
					double check = 1;
					while (check > 0.025) {
						NN.forwardProp(testsC[i][0]);
						NN.backProp(testsC[i][1]);
						double totalLoss = 0;
						for (int j = NEURALNETWORK.inputN+NEURALNETWORK.hiddenN; j < NEURALNETWORK.inputN+NEURALNETWORK.hiddenN+NEURALNETWORK.outputN; j++) {
							result[j-(NEURALNETWORK.inputN+NEURALNETWORK.hiddenN)] = NN.neurons[j].getOutput();
							double loss = Math.abs(result[j-(NEURALNETWORK.inputN+NEURALNETWORK.hiddenN)] - testsC[i][1][j-(NEURALNETWORK.inputN+NEURALNETWORK.hiddenN)]);
//							System.out.println(loss);
							totalLoss += loss;
						}
						check = totalLoss/3;
//						System.out.println(totalLoss/3);
						epochs += 1;
					}
					System.out.println("Done training at epoch " + epochs);
					epochs = 0;
					System.out.println("Target Output | Actual Output");
					for (int j = NEURALNETWORK.inputN+NEURALNETWORK.hiddenN; j < NEURALNETWORK.inputN+NEURALNETWORK.hiddenN+NEURALNETWORK.outputN; j++) {
							System.out.println(testsC[i][1][j-(NEURALNETWORK.inputN+NEURALNETWORK.hiddenN)]
											+ "           | " + NN.neurons[j].getOutput());
							result[j-(NEURALNETWORK.inputN+NEURALNETWORK.hiddenN)] = NN.neurons[j].getOutput();
					}
					double TSSE = NN.getTSSE(result);
					System.out.println("TSSE: " + TSSE);
					double RMSE = NN.getRMSE(TSSE);
					System.out.println("RMSE: "+ RMSE);
					
					System.out.println();
					System.out.println("Dirty results:");
					int minValInd;
					System.out.println("Target Output | Actual Output");
					NN.forwardProp(testsD[i][0]);
					//NN.backProp(testsD[i][1]);
					for (int j = NEURALNETWORK.inputN+NEURALNETWORK.hiddenN; j < NEURALNETWORK.inputN+NEURALNETWORK.hiddenN+NEURALNETWORK.outputN; j++) {
						System.out.println(testsD[i][1][j-(NEURALNETWORK.inputN+NEURALNETWORK.hiddenN)]
										+ "           | " + NN.neurons[j].getOutput());
						result[j-(NEURALNETWORK.inputN+NEURALNETWORK.hiddenN)] = NN.neurons[j].getOutput();
					}
					TSSE = NN.getTSSE(result);
					System.out.println("TSSE: " + TSSE);
					RMSE = NN.getRMSE(TSSE);
					System.out.println("RMSE: "+ RMSE);
					minValInd = NN.classify(result);
					if (minValInd == 0) {
						System.out.println("plus");}
					else if (minValInd == 1) {
						System.out.println("minus");}
					else if (minValInd == 2) {
						System.out.println("forward slash");}
					else if (minValInd == 3) {
						System.out.println("back slash");}
					else if (minValInd == 4) {
						System.out.println("x");}
					else if (minValInd == 5) {
						System.out.println("line");	}
					System.out.println();
				}
				//This needs to forward prop and compare its outputs to desired outputs to determine
				//which symbol the dirty set belongs to
				//Printing dirty results

//				for (int i = 0; i < testsD.length; i++) {
//					NN.forwardProp(testsD[i][0]);
//					NN.backProp(testsD[i][1]);
//					System.out.println("Target Output | Actual Output");
//					for (int j = NEURALNETWORK.inputN+NEURALNETWORK.hiddenN; j < NEURALNETWORK.inputN+NEURALNETWORK.hiddenN+NEURALNETWORK.outputN; j++) {
//						System.out.println(testsD[i][1][j-(NEURALNETWORK.inputN+NEURALNETWORK.hiddenN)]
//										+ "           | " + NN.neurons[j].getOutput());
//						result2[j-(NEURALNETWORK.inputN+NEURALNETWORK.hiddenN)] = NN.neurons[j].getOutput();
//					}
//					double TSSE = NN.getTSSE(result2);
//					System.out.println("TSSE: " + TSSE);
//					double RMSE = NN.getRMSE(TSSE);
//					System.out.println("RMSE: "+ RMSE);
//				}
				break;
//			case "runclean":
//				//Printing clean results
//				double[] result = {0,0,0};
//				System.out.println("Clean results:");
//				for (int i = 0; i < testsC.length; i++) {
//					NN.forwardProp(testsC[i][0]);
//					NN.backProp(testsC[i][0]);
//					System.out.println("Target Output | Actual Output");
//					for (int j = NEURALNETWORK.inputN+NEURALNETWORK.hiddenN; j < NEURALNETWORK.inputN+NEURALNETWORK.hiddenN+NEURALNETWORK.outputN; j++) {
//							System.out.println(testsC[i][1][j-(NEURALNETWORK.inputN+NEURALNETWORK.hiddenN)]
//											+ "           | " + NN.neurons[j].getOutput());
//							result[j-(NEURALNETWORK.inputN+NEURALNETWORK.hiddenN)] = NN.neurons[j].getOutput();
//					}
//					double TSSE = NN.getTSSE(result);
//					System.out.println("TSSE: " + TSSE);
//					double RMSE = NN.getRMSE(TSSE);
//					System.out.println("RMSE: "+ RMSE);
//				}
//				break;
			case "getdirty":
				// Displays dirty test set
				for (int i = 0; i < testsD.length; i++) {
					System.out.println(Arrays.deepToString(testsD[i]));
				}
				break;
			case "rundirty":
//				//This needs to forward prop and compare its outputs to desired outputs to determine
//				//which symbol the dirty set belongs to
//				//Printing dirty results
//				System.out.println("Dirty results:");
//				System.out.println("Noise level: " + noise);
//				double[] result2 = {0,0,0};
//				int minValInd;
//				for (int i = 0; i < testsD.length; i++) {
//					NN.forwardProp(testsD[i][0]);
//					NN.backProp(testsD[i][1]);
//					System.out.println("Target Output | Actual Output");
//					for (int j = NEURALNETWORK.inputN+NEURALNETWORK.hiddenN; j < NEURALNETWORK.inputN+NEURALNETWORK.hiddenN+NEURALNETWORK.outputN; j++) {
//						System.out.println(testsD[i][1][j-(NEURALNETWORK.inputN+NEURALNETWORK.hiddenN)]
//										+ "           | " + NN.neurons[j].getOutput());
//						result2[j-(NEURALNETWORK.inputN+NEURALNETWORK.hiddenN)] = NN.neurons[j].getOutput();
//					}
//					double TSSE = NN.getTSSE(result2);
//					System.out.println("TSSE: " + TSSE);
//					double RMSE = NN.getRMSE(TSSE);
//					System.out.println("RMSE: "+ RMSE);
//					minValInd = NN.classify(result2);
//					if (minValInd == 0) {
//						System.out.println("plus");}
//					else if (minValInd == 1) {
//						System.out.println("minus");}
//					else if (minValInd == 2) {
//						System.out.println("forward slash");}
//					else if (minValInd == 3) {
//						System.out.println("back slash");}
//					else if (minValInd == 4) {
//						System.out.println("x");}
//					else if (minValInd == 5) {
//						System.out.println("line");	}
//				}
				break;
			case "exit":
				System.exit(0);
			}
		}
	}
}
