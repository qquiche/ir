package ir.eval;

import java.io.*;
import java.util.*;
import java.lang.*;

import ir.utilities.*;
import ir.vsr.*;

/**
 * Evaluation experiment for testing relevance-rated feedback.
 * Simulates user feedback on top N documents and evaluates the revised query
 * on the residual corpus (after removing feedback documents).
 */
public class ExperimentRelFeedbackRated extends ExperimentRated {

    /**
     * Number of top documents to use for simulated feedback.
     */
    protected int numFeedbackDocs;

    /**
     * Whether to use binary feedback instead of rated feedback.
     */
    protected boolean useBinaryFeedback = false;

    /**
     * Whether to use control condition (no feedback, just remove top N docs).
     */
    protected boolean useControl = false;

    /**
     * Constructs an ExperimentRelFeedbackRated instance.
     *
     * @param corpusDir          The directory of files to index.
     * @param queryFile          The file of query/relevant-docs pairs to evaluate.
     * @param outFile            File for output precision/recall data.
     * @param docType            The type of documents to index.
     * @param stem               Whether to stem tokens.
     * @param numFeedbackDocs    Number of top documents for feedback.
     * @param useBinaryFeedback  Whether to use binary instead of rated feedback.
     * @param useControl         Whether to use control condition (no feedback).
     * @throws IOException if file I/O fails.
     */
    public ExperimentRelFeedbackRated(File corpusDir, File queryFile, File outFile, 
                                     short docType, boolean stem, int numFeedbackDocs,
                                     boolean useBinaryFeedback, boolean useControl)
        throws IOException {
        super(corpusDir, queryFile, outFile, docType, stem);
        this.numFeedbackDocs = numFeedbackDocs;
        this.useBinaryFeedback = useBinaryFeedback;
        this.useControl = useControl;
    }

    /**
     * Processes the next query with simulated relevance feedback.
     *
     * @param in BufferedReader for reading queries.
     * @return true if a query was successfully read; false if no more queries exist.
     * @throws IOException if reading fails.
     */
    boolean processQuery(BufferedReader in) throws IOException {
        String query = in.readLine();
        if (query == null) return false;
        
        System.out.println("\nQuery " + (rpResults.size() + 1) + ": " + query);

        Retrieval[] retrievals = index.retrieve(query);
        System.out.println("Returned " + retrievals.length + " documents.");

        ArrayList<String> correctRetrievals = new ArrayList<String>();
        getCorrectRatedRetrievals(in, correctRetrievals);
        
        System.out.println(correctRetrievals.size() + " truly relevant documents.");

        Set<String> feedbackDocs = simulateFeedback(query, retrievals, correctRetrievals);

        Retrieval[] finalRetrievals = retrievals;
        if (!useControl && feedbackDocs.size() > 0) {
            System.out.println("Executing New Expanded and Reweighted Query:");
            
            HashMapVector queryVector = (new TextStringDocument(query, index.stem)).hashMapVector();
            FeedbackRated fdback = new FeedbackRated(queryVector, retrievals, index);
            
            int numProcessed = Math.min(numFeedbackDocs, retrievals.length);
            for (int i = 0; i < numProcessed; i++) {
                String docName = retrievals[i].docRef.file.getName();
                
                if (correctRetrievals.contains(docName)) {
                    double rating = ratingsMap.get(docName);
                    if (useBinaryFeedback) {
                        fdback.addGood(retrievals[i].docRef, 1.0);
                    } else {
                        fdback.addGood(retrievals[i].docRef, rating);
                    }
                } else {
                    if (useBinaryFeedback) {
                        fdback.addBad(retrievals[i].docRef, -1.0);
                    } else {
                        fdback.addBad(retrievals[i].docRef, -1.0);
                    }
                }
            }
            
            HashMapVector newQuery = fdback.newQuery();
            finalRetrievals = index.retrieve(newQuery);
        }

        Retrieval[] residualRetrievals = removeDocuments(finalRetrievals, feedbackDocs);
        ArrayList<String> residualCorrect = new ArrayList<String>(correctRetrievals);
        residualCorrect.removeAll(feedbackDocs);

        rpResults.add(evalRetrievals(residualRetrievals, residualCorrect));
        UpdateNDCG(residualRetrievals, residualCorrect);

        String line = in.readLine();
        if (!(line == null || line.trim().equals(""))) {
            System.out.println("\nCould not find blank line after query, bad queryFile format");
            System.exit(1);
        }

        return true;
    }

    /**
     * Simulates user feedback on the top N documents using gold-standard relevance ratings.
     *
     * @param query             The original query string.
     * @param retrievals        The initial ranked retrievals.
     * @param correctRetrievals List of correct document names.
     * @return Set of document names that received feedback.
     */
    protected Set<String> simulateFeedback(String query, Retrieval[] retrievals,
                                          ArrayList<String> correctRetrievals) {
        Set<String> feedbackDocs = new HashSet<String>();
        List<String> positiveDocs = new ArrayList<String>();
        List<String> negativeDocs = new ArrayList<String>();

        System.out.println("\nFeedback:");

        int numToProcess = Math.min(numFeedbackDocs, retrievals.length);
        for (int i = 0; i < numToProcess; i++) {
            String docName = retrievals[i].docRef.file.getName();
            feedbackDocs.add(docName);

            if (correctRetrievals.contains(docName)) {
                positiveDocs.add(docName);
            } else {
                negativeDocs.add(docName);
            }
        }

        System.out.println("Positive docs: " + positiveDocs);
        System.out.println("Negative docs: " + negativeDocs);

        return feedbackDocs;
    }

    /**
     * Removes specified documents from a retrieval array.
     *
     * @param retrievals   Original retrieval array.
     * @param docsToRemove Set of document names to remove.
     * @return New retrieval array without the specified documents.
     */
    protected Retrieval[] removeDocuments(Retrieval[] retrievals, Set<String> docsToRemove) {
        List<Retrieval> filtered = new ArrayList<Retrieval>();
        
        for (Retrieval r : retrievals) {
            String docName = r.docRef.file.getName();
            if (!docsToRemove.contains(docName)) {
                filtered.add(r);
            }
        }
        
        return filtered.toArray(new Retrieval[filtered.size()]);
    }

    /**
     * Main entry point for running experiments.
     * <p>Usage:</p>
     * <pre>
     * ExperimentRelFeedbackRated [OPTIONS]* [DIR] [QUERIES] [OUTFILE] [N]
     * </pre>
     *
     * <p>Options:</p>
     * <ul>
     *   <li>-binary: Use binary feedback instead of rated feedback.</li>
     *   <li>-control: No feedback, just evaluate on residual corpus.</li>
     *   <li>-html: Process HTML documents.</li>
     *   <li>-stem: Use Porter stemmer.</li>
     * </ul>
     *
     * @param args Command-line arguments.
     * @throws IOException if any file operation fails.
     */
    public static void main(String[] args) throws IOException {
        boolean useBinaryFeedback = false;
        boolean useControl = false;
        short docType = DocumentIterator.TYPE_TEXT;
        boolean stem = false;
        
        int argIndex = 0;
        
        while (argIndex < args.length - 4) {
            String flag = args[argIndex];
            if (flag.equals("-binary")) {
                useBinaryFeedback = true;
            } else if (flag.equals("-control")) {
                useControl = true;
            } else if (flag.equals("-html")) {
                docType = DocumentIterator.TYPE_HTML;
            } else if (flag.equals("-stem")) {
                stem = true;
            } else {
                throw new IllegalArgumentException("Unknown flag: " + flag);
            }
            argIndex++;
        }

        String corpusDir = args[argIndex];
        String queryFile = args[argIndex + 1];
        String outFile = args[argIndex + 2];
        int numFeedbackDocs = Integer.parseInt(args[argIndex + 3]);

        ExperimentRelFeedbackRated exper = new ExperimentRelFeedbackRated(
            new File(corpusDir), new File(queryFile), new File(outFile),
            docType, stem, numFeedbackDocs, useBinaryFeedback, useControl);
        
        exper.makeRpCurve();
        exper.makeNDCGtable();
    }
}
