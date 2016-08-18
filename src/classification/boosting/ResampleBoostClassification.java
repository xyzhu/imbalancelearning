package classification.boosting;

import java.util.Random;

import classification.BasicClassification;
import evaluation.SampleEvaluation;
import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;

public class ResampleBoostClassification extends BasicClassification{

	public ResampleBoostClassification(Instances data) {
		super(data);
	}

	//get the classification result without bagging
	public String getClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = "";
		Random rand;
		//use different seed for 10-fold cross validation
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			AdaBoostM1 boost_classifier = new AdaBoostM1();
			boost_classifier.setClassifier(classifier);
			boost_classifier.setUseResampling(true);
			rand = new Random(randomSeed);
			SampleEvaluation eval = new SampleEvaluation(data);
			eval.crossValidateModel(boost_classifier, "oversample", data, 10, rand);//use 10-fold cross validataion
			predictResult = getName("oversampleboost", classifier_name);
			predictResult += getResult(eval);
		}
		
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			AdaBoostM1 boost_classifier = new AdaBoostM1();
			boost_classifier.setClassifier(classifier);
			boost_classifier.setUseResampling(true);
			rand = new Random(randomSeed);
			SampleEvaluation eval = new SampleEvaluation(data);
			eval.crossValidateModel(boost_classifier, "undersample", data, 10, rand);//use 10-fold cross validataion
			predictResult += getName(",undersampleboost", classifier_name);
			predictResult += getResult(eval);
		}
		return predictResult;
	}
}
