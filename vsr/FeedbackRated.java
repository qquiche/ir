package ir.vsr;

import java.util.*;
import java.io.*;
import ir.utilities.*;

/**
 * Extension of the Feedback class to handle continuous real-valued relevance ratings
 * instead of binary feedback. Ratings range from -1 (very irrelevant) to +1 (very relevant).
 */
public class FeedbackRated extends Feedback {

    /**
     * Stores continuous relevance ratings for documents.
     * Key: document reference, Value: rating (-1 to +1).
     */
    protected Map<DocumentReference, Double> docRatings;

    /**
     * Constructs a FeedbackRated instance.
     *
     * @param queryVector    The query vector representation.
     * @param retrievals     The initial set of document retrievals.
     * @param invertedIndex  The inverted index used for retrieval.
     */
    public FeedbackRated(HashMapVector queryVector, Retrieval[] retrievals, InvertedIndex invertedIndex) {
        super(queryVector, retrievals, invertedIndex);
        docRatings = new HashMap<DocumentReference, Double>();
    }

    /**
     * Adds a document to the list of relevant documents with a real-valued rating.
     * The rating indicates how relevant the document is (between 0 and 1).
     *
     * @param docRef The document reference to add.
     * @param rating The relevance rating (0 to 1, where 1 is most relevant).
     */
    public void addGood(DocumentReference docRef, double rating) {
        goodDocRefs.add(docRef);
        docRatings.put(docRef, rating);
    }

    /**
     * Adds a document to the list of non-relevant documents with a real-valued rating.
     * The rating indicates how irrelevant the document is (between -1 and 0).
     *
     * @param docRef The document reference to add.
     * @param rating The relevance rating (-1 to 0, where -1 is most irrelevant).
     */
    public void addBad(DocumentReference docRef, double rating) {
        badDocRefs.add(docRef);
        docRatings.put(docRef, rating);
    }

    /**
     * Generates a new query vector using the Ide Regular feedback algorithm
     * modified for continuous ratings.
     * <p>
     * The updated query is computed as:
     * <pre>
     * Q' = αQ + βΣ(r_i * D_i) - γΣ(|r_j| * D_j)
     * </pre>
     * where:
     * <ul>
     *   <li>Q is the original query vector</li>
     *   <li>D_i are relevant document vectors with rating r_i</li>
     *   <li>D_j are irrelevant document vectors with rating r_j</li>
     * </ul>
     *
     * @return The updated query vector incorporating feedback.
     */
    public HashMapVector newQuery() {
        HashMapVector newQuery = queryVector.copy();
        newQuery.multiply(ALPHA / newQuery.maxWeight());
        
        for (DocumentReference docRef : goodDocRefs) {
            Document doc = docRef.getDocument(invertedIndex.docType, invertedIndex.stem);
            HashMapVector vector = doc.hashMapVector();
            double rating = docRatings.get(docRef);
            vector.multiply(BETA * rating / vector.maxWeight());
            newQuery.add(vector);
        }
        
        for (DocumentReference docRef : badDocRefs) {
            Document doc = docRef.getDocument(invertedIndex.docType, invertedIndex.stem);
            HashMapVector vector = doc.hashMapVector();
            double rating = docRatings.get(docRef);
            vector.multiply(GAMMA * (-rating) / vector.maxWeight());
            newQuery.subtract(vector);
        }
        
        return newQuery;
    }

    /**
     * Retrieves the rating associated with a specific document.
     *
     * @param docRef The document reference.
     * @return The document's rating, or 0.0 if not found.
     */
    public double getRating(DocumentReference docRef) {
        Double rating = docRatings.get(docRef);
        return (rating != null) ? rating : 0.0;
    }
}
