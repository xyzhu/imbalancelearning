package classification;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;

import evaluation.OversampleEvaluation;
import evaluation.UndersampleEvaluation;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/*
 * This is the super class of other classification method
 */
public class BasicClassification {

	protected Instances data;
	DecimalFormat df;

	public BasicClassification(Instances data) {
		this.data = data;

	}


	public String classify(int times, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = getClassificationResult(classifier, classifier_name, times);//get the result without bagging
		return predictResult;
	}

	public Evaluation evaluate(Classifier classifier, int randomSeed, String sample)
			throws Exception {
		Random rand;
		rand = new Random(randomSeed);
		Evaluation eval = null;
		if(sample.equals("under")){
			eval = new UndersampleEvaluation(data);
			eval.crossValidateModel(classifier, data, 10, rand);
		}
		if(sample.equals("over")){
			eval = new OversampleEvaluation(data);
			eval.crossValidateModel(classifier, data, 10, rand);
		}
		else{
			eval = new Evaluation(data);
			eval.crossValidateModel(classifier, data, 10, rand);//use 10-fold cross validataion
		}
		return eval;
	}

	//save the interested result of the classification
	public String getResult(String methodname, String classifiername, double validationResult[], int times) throws Exception {
		df = (DecimalFormat) NumberFormat.getInstance();//use df to format result of be form of 0.0000
		df.applyPattern("0.0000");
		double accuracy = validationResult[0]/times;
		double recall_0 = validationResult[1]/times;
		double recall_1 = validationResult[2]/times;
		double precison_0 = validationResult[3]/times;
		double precison_1 = validationResult[4]/times;
		double fmeasure_0 = validationResult[5]/times;
		double fmeasure_1 = validationResult[6]/times;
		double gmean = validationResult[7]/times;
		double auc = validationResult[8]/times;
		return methodname + ", " + classifiername + ", " + df.format(accuracy) + ", " + df.format(recall_0) + ", " + df.format(recall_1)	+ ", "
		+ df.format(precison_0) + ", " + df.format(precison_1) + ", "+ df.format(fmeasure_0) + ", " 
		+ df.format(fmeasure_1) + "," + df.format(gmean) + "," + df.format(auc) + "\n";
	}

	public void updateResult(double validationResult[], Evaluation eval){
		double accuracy = eval.pctCorrect();
		double recall_0 = eval.recall(0);
		double recall_1 = eval.recall(1);
		double precison_0 = eval.precision(0);
		double precison_1 = eval.precision(1);
		double fmeasure_0 = eval.fMeasure(0);
		double fmeasure_1 = eval.fMeasure(1);
		double gmean = Math.sqrt(recall_0*recall_1);
		double auc = eval.areaUnderROC(0);
		validationResult[0] += accuracy;
		validationResult[1] += recall_0;
		validationResult[2] += recall_1;
		validationResult[3] += precison_0;
		validationResult[4] += precison_1;
		validationResult[5] += fmeasure_0;
		validationResult[6] += fmeasure_1;
		validationResult[7] += gmean;
		validationResult[8] += auc;
	}

	//save the interested result of the classification
	protected String getResultMatrix(Evaluation eval) throws Exception {
		return eval.toMatrixString() +"\n";
	}

	public String getClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception{
		return "";
	};
}
