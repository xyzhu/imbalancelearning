package main;

import java.io.BufferedReader;
import java.io.FileReader;

import classification.Classification;
import dataprocess.CmdLineParser;
import dataprocess.Util;
import weka.core.AttributeStats;
import weka.core.Instances;

public class Start {

	public static void main(String argv[]) throws Exception{
		/**
		 * Command line parsing
		 */
		CmdLineParser cmdparser = new CmdLineParser();
		CmdLineParser.Option dataset_opt = cmdparser.addStringOption('d', "dataset");
		CmdLineParser.Option filepath_opt = cmdparser.addStringOption('f', "filepath");
		CmdLineParser.Option classifier_opt = cmdparser.addStringOption('c', "classifier");

		try {
			cmdparser.parse(argv);
		} catch (CmdLineParser.OptionException e) {
			System.err.println(e.getMessage());
			//            printUsage();
			System.exit(2);
		}

		String dataset = (String) cmdparser.getOptionValue(dataset_opt, "weather");
		String filepath = (String) cmdparser.getOptionValue(filepath_opt, "");
		String classifier_name = (String) cmdparser.getOptionValue(classifier_opt,"j48");

		int numCrossValidations = 1;
		String datasets[] = {"breast-cancer"};
		Util util = new Util();
		String output_file = filepath + "result.txt";
		String measure_name = "dataset, method, classifier, accuracy, recall-0, recall-1, precision-0, precision-1, fMeasure-0, fMeasure-1, gmean, auc \n";
		util.saveResult(measure_name, output_file);
		String predict_result = "";
		int numDataset = datasets.length;
		for(int i=0;i<numDataset;i++){
			dataset = datasets[i];
			System.out.println(dataset);
			//read in the input arff file		
			String inputfile = filepath + dataset+".arff";

			FileReader fr = new FileReader(inputfile);
			BufferedReader br = new BufferedReader(fr);
			Instances data = new Instances(br);
			data.setClassIndex(data.numAttributes()-1);
			//print out number of instances
			System.out.println("Total number of instances: "+data.numInstances());
			AttributeStats as = data.attributeStats(data.numAttributes()-1);
			int count[] = as.nominalCounts;		
			System.out.println("Number of buggy instances: "+count[1]);
			Classification classification = new Classification(data);
			predict_result = classification.predict(classifier_name, filepath, dataset, numCrossValidations);
			util.appendResult(predict_result, output_file);
		}
	}

}
