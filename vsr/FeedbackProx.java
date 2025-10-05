package ir.vsr;

import java.util.*;
import ir.utilities.*;

/**
 * Relevance feedback helper for proximity-enhanced retrieval.
 *
 * Implements a Rocchio-style update to the original query vector using sets of
 * user-marked relevant (good) and non-relevant (bad) documents. The updated query is:
 * 
 *
 * 
 * Q_new = α · norm(Q_orig) + Σ (β · norm(D_good)) − Σ (γ · norm(D_bad))
 * 
 * where each vector is scaled by its maximum term weight before combining.
 */
public class FeedbackProx {

  /**
   * Weight for the original query component (α in Rocchio).
   * Can be adjusted prior to calling {@link #newQuery()}.
   */
  public static double ALPHA = 1;

  /**
   * Weight for relevant (positive) documents (β in Rocchio).
   * Can be adjusted prior to calling {@link #newQuery()}.
   */
  public static double BETA = 1;

  /**
   * Weight for non-relevant (negative) documents (γ in Rocchio).
   * Can be adjusted prior to calling {@link #newQuery()}.
   */
  public static double GAMMA = 1;

  /** The original query vector to be expanded/reweighted. */
  public HashMapVector queryVector;

  /** Retrieval results shown to the user (used for mapping indices to docs). */
  public Retrieval[] retrievals;

  /** Back-reference to the index for loading documents in the proper mode. */
  public InvertedPosIndex index;

  /** Set of user-marked relevant documents. */
  public ArrayList<DocumentReference> goodDocRefs = new ArrayList<>();

  /** Set of user-marked non-relevant documents. */
  public ArrayList<DocumentReference> badDocRefs = new ArrayList<>();

  /**
   * Creates a feedback session bound to an existing query, result list, and index.
   *
   * @param queryVector the original query vector
   * @param retrievals the retrieval results corresponding to the last query
   * @param index the index used to load documents for vectorization
   */
  public FeedbackProx(HashMapVector queryVector, Retrieval[] retrievals, InvertedPosIndex index) {
    this.queryVector = queryVector;
    this.retrievals = retrievals;
    this.index = index;
  }

  /**
   * Marks a document as relevant (positive feedback).
   *
   * @param docRef document reference to add
   */
  public void addGood(DocumentReference docRef) { goodDocRefs.add(docRef); }

  /**
   * Marks a document as non-relevant (negative feedback).
   *
   * @param docRef document reference to add
   */
  public void addBad(DocumentReference docRef) { badDocRefs.add(docRef); }

  /**
   * Indicates whether no feedback has been provided yet.
   *
   * @return {@code true} if both good and bad sets are empty; {@code false} otherwise
   */
  public boolean isEmpty() { return goodDocRefs.isEmpty() && badDocRefs.isEmpty(); }

  /**
   * Prompts the user for relevance feedback on a specific ranked document.
   * Accepts {@code y} (yes), {@code n} (no), or {@code u} (unsure).
   * Invalid responses cause the prompt to repeat.
   *
   * @param showNumber 1-based rank position in {@link #retrievals}
   */
  public void getFeedback(int showNumber) {
    DocumentReference docRef = retrievals[showNumber - 1].docRef;
    String response = UserInput.prompt("Is document #" + showNumber + ":" + docRef.file.getName() +
        " relevant (y:Yes, n:No, u:Unsure)?: ");
    if (response.equals("y"))
      goodDocRefs.add(docRef);
    else if (response.equals("n"))
      badDocRefs.add(docRef);
    else if (!response.equals("u"))
      getFeedback(showNumber);
  }

  /**
   * Checks whether feedback has already been recorded for a specific ranked document.
   *
   * @param showNumber 1-based rank position in {@link #retrievals}
   * @return {@code true} if the document is in either the good or bad lists; {@code false} otherwise
   */
  public boolean haveFeedback(int showNumber) {
    DocumentReference docRef = retrievals[showNumber - 1].docRef;
    return goodDocRefs.contains(docRef) || badDocRefs.contains(docRef);
  }

  /**
   * Generates a new, reweighted query vector using the current feedback.
   *
   * Each vector (original query, each good document, each bad document) is scaled by
   * its maximum component weight before combination to reduce sensitivity to length.
   *
   * @return the expanded and reweighted query vector
   */
  public HashMapVector newQuery() {
    HashMapVector newQuery = queryVector.copy();
    newQuery.multiply(ALPHA / newQuery.maxWeight());

    for (DocumentReference docRef : goodDocRefs) {
      Document doc = docRef.getDocument(index.docType, index.stem);
      HashMapVector vector = doc.hashMapVector();
      vector.multiply(BETA / vector.maxWeight());
      newQuery.add(vector);
    }
    for (DocumentReference docRef : badDocRefs) {
      Document doc = docRef.getDocument(index.docType, index.stem);
      HashMapVector vector = doc.hashMapVector();
      vector.multiply(GAMMA / vector.maxWeight());
      newQuery.subtract(vector);
    }
    return newQuery;
  }
}
