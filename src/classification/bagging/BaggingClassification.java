package classification.bagging;

import java.util.Random;

import classification.BasicClassification;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.core.Instances;

public class BaggingClassification extends BasicClassification{

	public BaggingClassification(Instances data) {
		super(data);
	}


	//using bagging classification method
	public String getClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = "";
		Random rand;
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			Bagging bag_classifier = new Bagging(); //set the classifier as bagging
			bag_classifier.setClassifier(classifier); //set the basic classifier of bagging
			rand = new Random(randomSeed);	
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(bag_classifier, data, 10, rand);//use 10-fold cross validataion
			predictResult = getName("bag", classifier_name);
			predictResult += getResult(eval);
		}
		return predictResult;
	}

}
