package classification.bagging;

import java.util.Random;

import classification.BasicClassification;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import Classifier.OverBagging;
import Classifier.SmoteBagging;
import Classifier.UnderBagging;
import Classifier.UnderOverBagging;

public class ResampleInBaggingClassification extends BasicClassification{

	public ResampleInBaggingClassification(Instances data) {
		super(data);
	}

	public String getClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception{
		String predictResult = "";
		predictResult += getOverBagClassificationResult(maxseed, classifier, classifier_name);
		predictResult += getUnderBagClassificationResult(maxseed, classifier, classifier_name);
		predictResult += getUnderOverBagClassificationResult(maxseed, classifier, classifier_name);
		//predictResult += getSmoteBagClassificationResult(maxseed, classifier, classifier_name);
		return predictResult;
	}

	private String getSmoteBagClassificationResult(int maxseed,
			Classifier classifier, String classifier_name) throws Exception {
		String predictResult = "";
		Random rand;
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			SmoteBagging bag_classifier = new SmoteBagging(); //set the classifier as bagging
			bag_classifier.setClassifier(classifier); //set the basic classifier of bagging
			rand = new Random(randomSeed);	
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(bag_classifier, data, 10, rand);//use 10-fold cross validataion
			predictResult = getName(",smotebag", classifier_name);
			predictResult += getResult(eval);
		}
		return predictResult;
	}

	//using bagging classification method with under sampling
	public String getUnderBagClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = "";
		Random rand;
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			UnderBagging bag_classifier = new UnderBagging(); //set the classifier as bagging
			bag_classifier.setClassifier(classifier); //set the basic classifier of bagging
			rand = new Random(randomSeed);	
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(bag_classifier, data, 10, rand);//use 10-fold cross validataion
			predictResult = getName(",underbag", classifier_name);
			predictResult += getResult(eval);
		}
		return predictResult;
	}


	private String getOverBagClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = "";
		Random rand;
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			OverBagging bag_classifier = new OverBagging(); //set the classifier as bagging
			bag_classifier.setClassifier(classifier); //set the basic classifier of bagging
			rand = new Random(randomSeed);	
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(bag_classifier, data, 10, rand);//use 10-fold cross validataion
			predictResult = getName("overbag", classifier_name);
			predictResult += getResult(eval);
		}
		return predictResult;
	}
	
	
	private String getUnderOverBagClassificationResult(int maxseed, Classifier classifier, String classifier_name) throws Exception {
		String predictResult = "";
		Random rand;
		for(int randomSeed = 1;randomSeed<=maxseed;randomSeed++){
			UnderOverBagging bag_classifier = new UnderOverBagging(); //set the classifier as bagging
			bag_classifier.setClassifier(classifier); //set the basic classifier of bagging
			rand = new Random(randomSeed);	
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(bag_classifier, data, 10, rand);//use 10-fold cross validataion
			predictResult = getName(",underoverbag", classifier_name);
			predictResult += getResult(eval);
		}
		return predictResult;
	}

}
