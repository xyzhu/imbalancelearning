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
		CmdLineParser.Option project_opt = cmdparser.addStringOption('p', "project");
		CmdLineParser.Option filepath_opt = cmdparser.addStringOption('f', "filepath");
		CmdLineParser.Option classifier_opt = cmdparser.addStringOption('c', "classifier");

		try {
			cmdparser.parse(argv);
		} catch (CmdLineParser.OptionException e) {
			System.err.println(e.getMessage());
			//            printUsage();
			System.exit(2);
		}

		String project = (String) cmdparser.getOptionValue(project_opt, "weather");
		String filepath = (String) cmdparser.getOptionValue(filepath_opt, "changebug\\");
		String classifier_name = (String) cmdparser.getOptionValue(classifier_opt,"j48");
 
		//change classification projects
		String projects[] = {"ant", "camel","itext","jedit","lucene","synapse","tomcat","voldemort"};
		
		//reopen bug projects
		//String projects[] = {"eclipse", "apache", "openoffice"};
		
		//test projects
		//String projects[] = {"weather"};
		Util util = new Util();
		String output_file = filepath + "result.txt";
		String measure_name = "project, method, classifier, accuracy, recall-0, recall-1, precision-0, precision-1, fMeasure-0, fMeasure-1, gmean, auc \n";
		util.saveResult(measure_name, output_file);
		String predict_result = "";
		for(int i=0;i<projects.length;i++){
			project = projects[i];
			System.out.println(project);
			//read in the input arff file		
			String inputfile = filepath + project+".arff";

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
			predict_result = classification.classify(classifier_name, filepath, project);
			util.appendResult(predict_result, output_file);
		}
	}

}
