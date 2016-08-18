package classification;

import java.text.DecimalFormat;
import java.text.NumberFormat;

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


	public String classify(int maxseed, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = getClassificationResult(maxseed, classifier, classifier_name);//get the result without bagging
		return predictResult;
	}
	
	public String getName(String methodname, String classifiername) {
		return methodname+", " + classifiername + ", ";
	}

	//save the interested result of the classification
	public String getResult(Evaluation eval) throws Exception {
		df = (DecimalFormat) NumberFormat.getInstance();//use df to format result of be form of 0.0000
		df.applyPattern("0.0000");
		double accuracy = eval.pctCorrect();
		double recall_0 = eval.recall(0);
		double recall_1 = eval.recall(1);
		double precison_0 = eval.precision(0);
		double precison_1 = eval.precision(1);
		double fmeasure_0 = eval.fMeasure(0);
		double fmeasure_1 = eval.fMeasure(1);
		double gmean = Math.sqrt(recall_0*recall_1);
		double auc = eval.areaUnderROC(0);
		return df.format(accuracy) + ", " + df.format(recall_0) + ", " + df.format(recall_1)	+ ", "
		+ df.format(precison_0) + ", " + df.format(precison_1) + ", "+ df.format(fmeasure_0) + ", " 
		+ df.format(fmeasure_1) + "," + df.format(gmean) + "," + df.format(auc) + "\n";
	}

	//save the interested result of the classification
	protected String getResultMatrix(Evaluation eval) throws Exception {
		return eval.toMatrixString() +"\n";
	}

	public String getClassificationResult(int numseed, Classifier classifier, String classifier_name) throws Exception{
		return "";
		};
}
