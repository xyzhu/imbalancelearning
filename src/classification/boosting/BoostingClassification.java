package classification.boosting;

import java.util.Random;

import classification.BasicClassification;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;

public class BoostingClassification extends BasicClassification{

	public BoostingClassification(Instances data) {
		super(data);
	}


	//using bagging classification method
	public String getClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = "";
		Random rand;
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			AdaBoostM1 boost_classifier = new AdaBoostM1(); //set the classifier as bagging
			boost_classifier.setClassifier(classifier); //set the basic classifier of bagging
			rand = new Random(randomSeed);	
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(boost_classifier, data, 10, rand);//use 10-fold cross validataion
			predictResult = getName("boost", classifier_name);
			predictResult += getResult(eval);
		}
		return predictResult;
	}

}
