package classification.boosting;

import classification.BasicClassification;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import Classifier.OverBoosting;
import Classifier.UnderBoosting;

public class ResampleInBoostingClassification extends BasicClassification{

	public ResampleInBoostingClassification(Instances data) {
		super(data);
	}

	public String getClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception{
		String predictResult = "";
		predictResult = getOverBoostClassificationResult(classifier, classifier_name, times);
		predictResult += getUnderBoostClassificationResult(classifier, classifier_name, times);
		//predictResult += getSmoteBoostClassificationResult(maxseed, classifier, classifier_name, times);
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
	public String getUnderBoostClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult[] = new double[9];
		UnderBoosting boost_classifier = new UnderBoosting(); //set the classifier as bagging
		boost_classifier.setClassifier(classifier); //set the basic classifier of bagging
		boost_classifier.setUseResampling(true);
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(boost_classifier, randomSeed, "none");
			updateResult(validationResult, eval);
		}
		return getResult(",underinboost", classifier_name, validationResult, times);
	}


	private String getOverBoostClassificationResult(Classifier classifier, String classifier_name, int times) throws Exception {
		double validationResult[] = new double[9];
		OverBoosting boost_classifier = new OverBoosting(); //set the classifier as bagging
		boost_classifier.setClassifier(classifier); //set the basic classifier of bagging
		boost_classifier.setUseResampling(true);
		for(int randomSeed = 1;randomSeed<=times;randomSeed++){
			Evaluation eval = evaluate(boost_classifier, randomSeed, "none");
			updateResult(validationResult, eval);
		}
		return getResult("overinboost", classifier_name, validationResult, times);
	}

}
