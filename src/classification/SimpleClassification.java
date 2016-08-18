package classification;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class SimpleClassification extends BasicClassification{

	public SimpleClassification(Instances data) {
		super(data);
	}

	//get the classification result without bagging
	public String getClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = "";
		Random rand;
		//use different seed for 10-fold cross validation
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			rand = new Random(randomSeed);
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(classifier, data, 10, rand);//use 10-fold cross validataion
			predictResult = getName("simple", classifier_name);
			predictResult += getResult(eval);
		}
		return predictResult;
	}
}
