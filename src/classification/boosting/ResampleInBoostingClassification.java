package classification.boosting;

import java.util.Random;

import classification.BasicClassification;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import Classifier.OverBagging;
import Classifier.OverBoosting;
import Classifier.OverWeightBoosting;
import Classifier.SmoteBagging;
import Classifier.UnderBagging;
import Classifier.UnderBoosting;
import Classifier.UnderWeightBoosting;

public class ResampleInBoostingClassification extends BasicClassification{

	public ResampleInBoostingClassification(Instances data) {
		super(data);
	}

	public String getClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception{
		String predictResult = "";
		predictResult = getOverBoostClassificationResult(maxseed, classifier, classifier_name);
		predictResult += getUnderBoostClassificationResult(maxseed, classifier, classifier_name);
		//predictResult += getSmoteBoostClassificationResult(maxseed, classifier, classifier_name);
		predictResult += getOverWeightBoostClassificationResult(maxseed, classifier, classifier_name);
		predictResult += getUnderWeightBoostClassificationResult(maxseed, classifier, classifier_name);
		return predictResult;
	}

	//	private String getSmoteBoostClassificationResult(int maxseed,
	//			Classifier classifier, String classifier_name) throws Exception {
	//		String predictResult = "";
	//		Random rand;
	//		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
	//			SmoteBoosting boost_classifier = new SmoteBoosting(); //set the classifier as bagging
	//			boost_classifier.setClassifier(classifier); //set the basic classifier of bagging
	//			rand = new Random(randomSeed);	
	//			Evaluation eval = new Evaluation(data);
	//			eval.crossValidateModel(boost_classifier, data, 10, rand);//use 10-fold cross validataion
	//			predictResult = getName(",smoteboost", classifier_name);
	//			predictResult += getResult(eval);
	//		}
	//		return predictResult;
	//	}

	//using bagging classification method with under sampling
	public String getUnderBoostClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = "";
		Random rand;
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			UnderBoosting boost_classifier = new UnderBoosting(); //set the classifier as bagging
			boost_classifier.setClassifier(classifier); //set the basic classifier of bagging
			boost_classifier.setUseResampling(true);
			rand = new Random(randomSeed);	
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(boost_classifier, data, 10, rand);//use 10-fold cross validataion
			predictResult = getName(",underboost", classifier_name);
			predictResult += getResult(eval);
		}
		return predictResult;
	}


	private String getOverBoostClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = "";
		Random rand;
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			OverBoosting boost_classifier = new OverBoosting(); //set the classifier as bagging
			boost_classifier.setClassifier(classifier); //set the basic classifier of bagging
			boost_classifier.setUseResampling(true);
			rand = new Random(randomSeed);	
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(boost_classifier, data, 10, rand);//use 10-fold cross validataion
			predictResult = getName("overboost", classifier_name);
			predictResult += getResult(eval);
		}
		return predictResult;
	}

	private String getOverWeightBoostClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = "";
		Random rand;
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			OverWeightBoosting boost_classifier = new OverWeightBoosting(); //set the classifier as bagging
			boost_classifier.setClassifier(classifier); //set the basic classifier of bagging
			boost_classifier.setUseResampling(false);
			rand = new Random(randomSeed);	
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(boost_classifier, data, 10, rand);//use 10-fold cross validataion
			predictResult = getName(",overweightboost", classifier_name);
			predictResult += getResult(eval);
		}
		return predictResult;
	}
	
	private String getUnderWeightBoostClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = "";
		Random rand;
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			UnderWeightBoosting boost_classifier = new UnderWeightBoosting(); //set the classifier as bagging
			boost_classifier.setClassifier(classifier); //set the basic classifier of bagging
			boost_classifier.setUseResampling(false);
			rand = new Random(randomSeed);	
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(boost_classifier, data, 10, rand);//use 10-fold cross validataion
			predictResult = getName(",underweightboost", classifier_name);
			predictResult += getResult(eval);
		}
		return predictResult;
	}

}
