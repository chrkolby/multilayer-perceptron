import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.Locale;
import java.util.*;
import java.util.Arrays;


class Multilayer{

	public static void main(String[] args) throws IOException {
	
		String filename = "movements_day1-3.dat";
		
		FileReader file = new FileReader(filename);
		
		int counter = 0;
			
		Scanner sc = new Scanner(file).useLocale(Locale.US);
	

		while(sc.hasNext()){
				
			String word = sc.nextLine();

			counter++;
				
		}
		sc.close();
		
		FileReader reFile = new FileReader(filename);

		sc = new Scanner(reFile).useLocale(Locale.US);
		
		float[][] movements = new float[counter][41];
			
		int i = 0;
		int j = 0;

		while(sc.hasNext()){
			
			float number = sc.nextFloat();
			
				
			if(j == 41){
				i++;
				j = 0;
			}
			
				
			movements[i][j] = number;
			j++;
		}
		sc.close();
		
		Calculations ca = new Calculations(movements, counter);
		
		movements = ca.getMove();
	}
}

class Mlp{
	
	private float beta;
	private float eta;
	private float momentum;
	private float[][] train;
	private int[][] traint;
	private int hidden;
	private float[][] weightIn;
	private float[][] weightOut;
	private float[] inputLayer;
	private float[] hiddenLayer;
	private float[][] test;
	float[] errorIn;
	float[] errorOut;
	
	public Mlp(float[][] inputs, int[][] targets, int nhidden){
		
		beta = (float)1;
		eta = (float)0.1;
		momentum = (float)0.1;
		train = inputs;
		traint = targets;
		hidden = nhidden;
		errorIn = new float[hidden+1];
		errorOut = new float[8];
		hiddenLayer = new float[nhidden+1];
		inputLayer = new float[41];
		weightIn = new float[41][hidden+1];
		weightOut = new float[hidden+1][8];
		test = new float[8][inputs.length];
		createweights();
		
		
	}
	
	public int[] shuffle(int l){
		
		int[] shuffle = new int[l];
		
		for(int i = 0; i < l; i++){
			
			shuffle[i] = i;
			
		}
		
		int index;
		Random random = new Random();
		for (int j = shuffle.length - 1; j > 0; j--){
			index = random.nextInt(j + 1);
			if (index != j){
				shuffle[index] ^= shuffle[j];
				shuffle[j] ^= shuffle[index];
				shuffle[index] ^= shuffle[j];
			}
		}
		return shuffle;
	}
	
	public void createweights(){
		
		for(int i = 0; i<=40; i++){
			for(int j = 0; j<=hidden; j++){
				
				weightIn[i][j] = (float)(Math.random()-0.5);
				
			}
		}
		
		for(int i = 0; i<=hidden; i++){
			for(int j = 0; j<8; j++){
				
				weightOut[i][j] = (float)(Math.random()-0.5);
				
			}
		}
	}
	
	
	public void earlystopping(float[][] inputs, int[][] targets, float[][] valid, int[][] validt){
		
		boolean stop = false;
		float [][] validErrO = new float[valid.length][errorOut.length];
		float [][] outPrev = new float[valid.length][errorOut.length];;
		for(int i = 0; i<valid.length; i++){
		Arrays.fill(outPrev[i],1);
		}
		int t = 0;
		while(stop == false){
			
			//run training 100 times
			for(int j = 0; j<100; j++){
				int[] shuffle = shuffle(inputs.length);
				for(int i = 0; i<inputs.length; i++){
					
					train(inputs[shuffle[i]],targets[shuffle[i]]);
					
				}
			}
			
			//run the validation set and check if the error has gone up
			for(int i = 0; i<valid.length; i++){
				float[] use = train(valid[i],validt[i]);
				for(int j = 0; j<use.length; j++){
					
					validErrO[i][j] = Math.abs(use[j]);
					
				}
			}
			
			float allErrO = 0;
			float allErrOPrev = 0;
			for(int i = 0; i<valid.length; i++){
				for(int j = 0; j<errorOut.length; j++){
					allErrO += validErrO[i][j];
					allErrOPrev += outPrev[i][j];
					
				}
			}
			allErrO += -0.5;
			if(allErrO > allErrOPrev && t > 100){
				stop = true;
				
			}
			if(t>200)
				stop = true;
			for(int i = 0; i<valid.length; i++){
				for(int j = 0; j<errorOut.length; j++){
					outPrev[i][j] = validErrO[i][j];
				}
			}
			System.out.println(t);
			t++;
		}
	}
	
	public float[] train(float[] inputs, int[] targets){
		
			//find output error
			float[] out = forward(inputs);	
			for(int i = 0; i<8; i++){
				errorOut[i] = 0;
				errorOut[i] = out[i]*(1-out[i])*(targets[i]-out[i]);
			}
			//find input error
			for(int i = 0; i<=hidden; i++){
				float temp = 0;
				for(int j = 0; j<8; j++){
					
					temp += weightOut[i][j] * errorOut[j];
					
				}
				errorIn[i] = 0;
				errorIn[i] = hiddenLayer[i] * (1-hiddenLayer[i])*temp;

			}
			//adjust output weight
			for(int i = 0; i < 8; i++){
				
				for(int j = 0; j<=hidden; j++){
					weightOut[j][i] += (eta * errorOut[i] * hiddenLayer[j]);
				}
				
			}
			//adjust input weight
			for(int i = 0; i <= hidden; i++){
				
				for(int j = 0; j<=40; j++){
					weightIn[j][i] += (eta * errorIn[i] * inputLayer[j]);
				}
			}
			return errorOut;
		
	
	}
	
		

	
	public float[] forward(float[] inputs){
		
		float[] out = new float[8];
		
		for(int i = 0; i<40; i++){
			
			inputLayer[i+1] = inputs[i];
			
		}
		
		inputLayer[0] = -1;
		hiddenLayer[0] = -1;
		
		//fill the hidden layer
		for(int i = 0; i<=hidden; i++){
			
			hiddenLayer[i] = 0;
			
			
			for(int j = 0; j<=40; j++){
				
				hiddenLayer[i] += weightIn[j][i]*inputLayer[j];
				
			}
			float temp = 0;
			for(int x = 0; x<hidden; x++){
				temp += Math.exp(hiddenLayer[x]);
			}
			hiddenLayer[i] = (float)(1.0/(1.0+Math.exp(-hiddenLayer[i])));
		}
		
		//fill the output layer
		for(int i = 0; i<8; i++){
			
			out[i] = 0;
			
			for(int j = 0; j<=hidden; j++){
				
				out[i] += weightOut[j][i]*hiddenLayer[j];
				
			}
			float temp = 0;
			for(int x = 0; x<8; x++){
				temp += Math.exp(out[x]);
			}
			out[i] = (float)(1.0/(1.0+Math.exp(-out[i])));


		}

		return out;
	}
	
	public void confusion(float[][] inputs, int[][] targets){
		
		int[][] result = new int[8][8];
		int[][] temp = new int[8][8];
		int correct = 0;
		for(int i = 0; i<8; i++){
			
			Arrays.fill(result[i],0);
			
		}
		
		for(int i = 0; i<inputs.length; i++){
			
			float[] output = forward(inputs[i]);
			
			float max = 0;
			int x = 0;
			int h = 0;
			
			for(int j = 0; j<output.length; j++){
				
				
				if(max < output[j]){
					
					max = output[j];
					x = j;
					
				}
				if(targets[i][j] == 1){
					
					h=j;
					
				}
			}
			result[x][x] += 1;
			temp[h][h]++;
		
			
		}
		for(int i = 0; i<8; i++){
			for(int j = 0; j<8; j++){
				
				
				System.out.print(temp[i][j] + " ");
				
			}
			System.out.println("");
						
		}
		for(int i = 0; i<8; i++){
			for(int j=0; j<8; j++){
				
				System.out.print(result[i][j] + " ");
				
			}
			System.out.println("");
			correct = correct + Math.abs(result[i][i]-temp[i][i]); 

		}
		correct = inputs.length-correct;
		
		double percent = ((double)correct/(double)inputs.length)*100;
	
		
		System.out.println("You got " + percent + "% correct.");

		
	}
	
}
//creates the arrays and initialize the mlp class
class Calculations{
	
	private float[][] movements;
	private int[][] target;
	private int[] shuffle;
	private int rows;
	
	public Calculations(float[][] m, int r){
		
		movements = m;
		rows = r;
		shuffle = new int[rows];
		target = new int[rows][8];
		float[] avg = new float[40];
		avg = findAvg();
		subtract(avg);
		float[] imax = new float[40];
		imax = findMax();
		divide(imax);
		shuffle();
		createTarget();
		float[][] train = createTrain();
		int[][] traint = createTraint();
		float[][] valid = createValid();
		int[][] validt = createValidt();
		float[][] test = createTest();
		int[][] testt = createTestt();
		
		Mlp mlp = new Mlp(train, traint, 12);
		mlp.earlystopping(train,traint,valid,validt);
		mlp.confusion(test,testt);
		
		
	}
	
	
	public void printShuffle(){
		
		for(int i = 0; i<rows; i++){
			System.out.println(shuffle[i]);
		}
		
	}
	
	public void shuffle(){
		
		for(int i = 0; i < rows; i++){
			
			shuffle[i] = i;
			
		}
		
		int index;
		Random random = new Random();
		for (int j = shuffle.length - 1; j > 0; j--){
			index = random.nextInt(j + 1);
			if (index != j){
				shuffle[index] ^= shuffle[j];
				shuffle[j] ^= shuffle[index];
				shuffle[index] ^= shuffle[j];
			}
		}
	}
	
	public void printValid(float[][] valid){
		
		for(int i = 0; i<valid.length; i++){
			
			for(int j = 0; j<40; j++){
				
				
			}
			System.out.println(valid[i][0] + "  " +i);
			
		}		
	}
	
	public float[][] createValid(){
		
		float[][] valid = new float[rows/4][40];
		
		int x = 1;
		
		for(int i = 0; i<valid.length; i++){
			
			for(int j = 0; j<40; j++){
			
				valid[i][j] = movements[shuffle[x]][j];
			
			}
			
			x = x+4;
			
		}
		return valid;
		
	}
	
	public float[][] createTest(){
		
		float[][] test = new float[rows/4][40];
		
		int x = 3;
		
		for(int i = 0; i<test.length; i++){
			
			for(int j = 0; j<40; j++){
			
				test[i][j] = movements[shuffle[x]][j];
			
			}
			
			x = x+4;
			
		}
		return test;
		
	}
	
	public int[][] createTestt(){
		
		int[][] testt = new int[rows/4][8];
		
		int x = 1;
		
		for(int i = 0; i<testt.length; i++){
			
			for(int j = 0; j<8; j++){			
				testt[i][j] = target[x][j];			
			}
			x = x+4;
		}
		return testt;	
	}
	
	
	public float[][] createTrain(){
		
		float[][] train = new float[rows/2][40];
		
		int x = 0;
		
		for(int i = 0; i<train.length; i++){
			
			for(int j = 0; j<40; j++){
			
				train[i][j] = movements[shuffle[x]][j];
			
			}
			
			x = x+2;
			
		}
		return train;
		
	}
	
	public void printTraint(int[][] traint){
		
		for(int i = 0; i<traint.length; i++){
			
			for(int j = 0; j<8; j++){
				
			System.out.print(traint[i][j] + "  ");
				
			}
			System.out.println("");
			
		}		
	}
	
	public int[][] createValidt(){
		
		int[][] validt = new int[rows/4][8];
		
		int x = 1;
		
		for(int i = 0; i<validt.length; i++){
			
			for(int j = 0; j<8; j++){			
				validt[i][j] = target[x][j];			
			}
			x = x+4;
		}
		return validt;	
	}
	
	public void printValidt(int[][] validt){
		
		for(int i = 0; i<validt.length; i++){
			
			for(int j = 0; j<8; j++){
				
			System.out.print(validt[i][j] + "  ");
				
			}
			System.out.println("");
			
		}		
	}
	
	public int[][] createTraint(){
		
		int[][] traint = new int[rows/2][8];
		
		int x = 0;
		
		for(int i = 0; i<traint.length; i++){
			
			for(int j = 0; j<8; j++){			
				traint[i][j] = target[x][j];			
			}
			x = x+2;
		}
		return traint;	
	}
	
	public void printTarget(){
		
		for(int i = 0; i<rows; i++){
			
			for(int j = 0; j<8; j++){
				
				System.out.print(target[i][j] + "  ");
				
			}
			System.out.println("");
			
		}
		
	}
	
	public void createTarget(){
		
		float ii = 1;
		
		for(int i = 0; i<rows; i++){
			
			for(int j = 0; j<41; j++){
				
				
				if(movements[i][40] != ii){
					ii++;
					if(ii == 9)
						ii = 1;
				}
				
	
				movements[i][40] = ii;
				
				
				
				if(j == 40){
					
					target[shuffle[i]][(int)ii-1] = 1;
					
				}
				
			}
		}
		
	}
	
	public float[][] getMove (){
		
		return movements;
		
	}
	
	public void printMove(){	
		for(int x = 0; x<40; x++){
			
		
			
			for(int y = 0; y<rows; y++){
				System.out.println(movements[y][x]);
			}
		}
				
	}
	
	public float[] findAvg(){
		float[] average = new float[40];
		for(int x = 0; x<40; x++){
			
			for(int y = 0; y<rows; y++){
				
				average[x] = movements[y][x] + average[x];

				
			}
			average[x] = (average[x]/rows);
			
		}
		return average;
	}
	
	public void subtract(float[] avg){
		
		for(int i = 0; i<40; i++){
			
			for(int j = 0; j<rows; j++){
				
				movements[j][i] = movements[j][i] - avg[i];
				
			}
			
		}
		
	}
	
	public float[] findMax(){
		
		float[]imax = new float[40];
		
		for(int i = 0; i<40; i++){
			
			for(int j = 0; j<rows; j++){
				
				if(imax[i] < movements[j][i])
					imax[i] = movements[j][i];
				
				if(imax[i] < Math.abs(movements[j][i]))
					imax[i] = Math.abs(movements[j][i]);
				
				
			}
		}
		return imax;
	}
	
	public void divide (float[]imax){
		
		for(int i = 0; i<40; i++){
			
			
			for(int j = 0; j<rows; j++){
				
				movements[j][i] = (movements[j][i]/imax[i]);
			}
		}
	}
}