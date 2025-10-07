package ir.vsr;

import java.io.*;
import java.util.*;
import ir.utilities.*;

/**
 * Extension of the InvertedIndex class to support continuous relevance-rated feedback.
 * Allows users to provide real-valued ratings between -1 (very irrelevant)
 * and +1 (very relevant).
 */
public class InvertedIndexRated extends InvertedIndex {

    /**
     * Creates an inverted index of the documents in a directory.
     *
     * @param dirFile   A directory of document files to be indexed.
     * @param docType   The type of documents to index (see docType in DocumentIterator).
     * @param stem      Whether to stem tokens using the Porter stemmer.
     * @param feedback  Whether to enable relevance feedback.
     */
    public InvertedIndexRated(File dirFile, short docType, boolean stem, boolean feedback) {
        super(dirFile, docType, stem, feedback);
    }

    /**
     * Presents retrievals with support for rated feedback.
     * Replaces binary feedback with {@link FeedbackRated} to incorporate real-valued ratings.
     *
     * @param queryVector The query vector representation.
     * @param retrievals  The retrieval results to display.
     */
    public void presentRetrievals(HashMapVector queryVector, Retrieval[] retrievals) {
        if (showRetrievals(retrievals)) {
            FeedbackRated fdback = null;
            if (feedback)
                fdback = new FeedbackRated(queryVector, retrievals, this);
            
            int currentPosition = MAX_RETRIEVALS;
            int showNumber = 0;
            
            do {
                String command = UserInput.prompt("\n Enter command:  ");
                if (command.equals(""))
                    break;
                    
                if (command.equals("m")) {
                    printRetrievals(retrievals, currentPosition);
                    currentPosition = currentPosition + MAX_RETRIEVALS;
                    continue;
                }
                
                if (command.equals("r") && feedback) {
                    if (fdback.isEmpty()) {
                        System.out.println("Need to first view some documents and provide feedback.");
                        continue;
                    }
                    System.out.println("Positive docs: " + fdback.goodDocRefs +
                        "\nNegative docs: " + fdback.badDocRefs);
                    System.out.println("Executing New Expanded and Reweighted Query: ");
                    queryVector = fdback.newQuery();
                    retrievals = retrieve(queryVector);
                    fdback.retrievals = retrievals;
                    if (showRetrievals(retrievals))
                        continue;
                    else
                        break;
                }
                
                try {
                    showNumber = Integer.parseInt(command);
                } catch (NumberFormatException e) {
                    System.out.println("Unknown command.");
                    System.out.println("Enter `m' to see more, a number to show the nth document, nothing to exit.");
                    if (feedback && !fdback.isEmpty())
                        System.out.println("Enter `r' to use any feedback given to `redo' with a revised query.");
                    continue;
                }
                
                if (showNumber > 0 && showNumber <= retrievals.length) {
                    System.out.println("Showing document " + showNumber + " in the " + Browser.BROWSER_NAME + " window.");
                    Browser.display(retrievals[showNumber - 1].docRef.file);
                    if (feedback && !fdback.haveFeedback(showNumber))
                        getFeedbackRated(retrievals[showNumber - 1], fdback);
                } else {
                    System.out.println("No such document number: " + showNumber);
                }
            } while (true);
        }
    }

    /**
     * Prompts the user for a real-valued feedback rating between -1 and +1
     * for a given document retrieval and records it.
     *
     * @param retrieval The retrieval being rated.
     * @param fdback    The {@link FeedbackRated} object used to store ratings.
     */
    protected void getFeedbackRated(Retrieval retrieval, FeedbackRated fdback) {
        String response = UserInput.prompt("Enter relevance rating for this document (-1 to +1, 0 for not relevant): ");
        
        try {
            double rating = Double.parseDouble(response.trim());
            
            if (rating < -1.0 || rating > 1.0) {
                System.out.println("Invalid rating. Must be between -1 and +1. Using 0.");
                rating = 0.0;
            }
            
            if (rating > 0) {
                fdback.addGood(retrieval.docRef, rating);
                System.out.println("Added to relevant documents with rating: " + rating);
            } else if (rating < 0) {
                fdback.addBad(retrieval.docRef, rating);
                System.out.println("Added to irrelevant documents with rating: " + rating);
            } else {
                System.out.println("No feedback recorded for this document.");
            }
        } catch (NumberFormatException e) {
            System.out.println("Invalid number format. No feedback recorded.");
        }
    }

    /**
     * Indexes a directory of files and then interactively accepts retrieval queries
     * with rated feedback support.
     * <p>
     * Usage:
     * <pre>
     * InvertedIndexRated [OPTION]* [DIR]
     * </pre>
     * <p>
     * Options:
     * <ul>
     *   <li>-html: Process HTML files and remove tags.</li>
     *   <li>-stem: Stem tokens using the Porter stemmer.</li>
     *   <li>-feedback: Enable rated relevance feedback interaction.</li>
     * </ul>
     *
     * @param args Command-line arguments.
     */
    public static void main(String[] args) {
        String dirName = args[args.length - 1];
        short docType = DocumentIterator.TYPE_TEXT;
        boolean stem = false, feedback = false;
        
        for (int i = 0; i < args.length - 1; i++) {
            String flag = args[i];
            if (flag.equals("-html"))
                docType = DocumentIterator.TYPE_HTML;
            else if (flag.equals("-stem"))
                stem = true;
            else if (flag.equals("-feedback"))
                feedback = true;
            else {
                throw new IllegalArgumentException("Unknown flag: " + flag);
            }
        }

        InvertedIndexRated index = new InvertedIndexRated(new File(dirName), docType, stem, feedback);
        index.processQueries();
    }
}
