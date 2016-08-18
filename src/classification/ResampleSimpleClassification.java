package classification;

import java.util.Random;

import evaluation.SampleEvaluation;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class ResampleSimpleClassification extends BasicClassification{

	public ResampleSimpleClassification(Instances data) {
		super(data);
	}

	//get the classification result without bagging
	public String getClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = "";
		Random rand;
		//use different seed for 10-fold cross validation
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			rand = new Random(randomSeed);
			SampleEvaluation eval = new SampleEvaluation(data);
			eval.crossValidateModel(classifier, "oversample", data, 10, rand);//use 10-fold cross validataion
			predictResult = getName("oversample", classifier_name);
			predictResult += getResult(eval);
		}
		
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			rand = new Random(randomSeed);
			SampleEvaluation eval = new SampleEvaluation(data);
			eval.crossValidateModel(classifier, "undersample", data, 10, rand);//use 10-fold cross validataion
			predictResult += getName(",undersample", classifier_name);
			predictResult += getResult(eval);
		}
		return predictResult;
	}
}
