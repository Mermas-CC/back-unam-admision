import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import chromadb

# Cargar configuración
load_dotenv()
PERSIST_DIR = "./chroma_db_v2"

def main():
    print("📊 [SIMILARITY AUDIT] Validando discriminación de embeddings...")
    
    if not os.path.exists(PERSIST_DIR):
        print(f"❌ Error: No se encontró la base de datos en {PERSIST_DIR}")
        return

    db = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        collection = db.get_collection("admision_unap")
    except:
        print("❌ Error: Colección 'admision_unap' no encontrada.")
        return
        
    results = collection.get(include=['embeddings', 'metadatas'])
    embeddings = np.array(results['embeddings'])
    metadatas = results['metadatas']
    
    if len(embeddings) < 2:
        print("❌ Datos insuficientes para auditar.")
        return

    # Mapear chunks a documentos
    doc_map = {}
    for i, m in enumerate(metadatas):
        fname = m.get('file_name', 'unknown')
        doc_map.setdefault(fname, []).append(i)

    # Similitud Coseno Global
    sim_matrix = cosine_similarity(embeddings)
    
    intra_sims, inter_sims = [], []
    
    for doc, indices in doc_map.items():
        if len(indices) < 2: continue
        
        # Intra: Similitud entre chunks del mismo doc
        m_intra = sim_matrix[np.ix_(indices, indices)]
        intra_sims.extend(m_intra[np.triu_indices_from(m_intra, k=1)])
        
        # Inter: Similitud con el resto de docs
        others = [i for i in range(len(embeddings)) if i not in indices]
        if others:
            inter_sims.extend(sim_matrix[np.ix_(indices, others)].flatten())

    m_intra = np.mean(intra_sims) if intra_sims else 0
    m_inter = np.mean(inter_sims) if inter_sims else 0
    
    print(f"\n📈 RESULTADOS:")
    print(f"🔹 Similitud Intra-Documento: {m_intra:.4f}")
    print(f"🔸 Similitud Inter-Documento: {m_inter:.4f}")
    print(f"🚀 Ratio de Separación: {(m_intra - m_inter):.4f}")
    
    if m_intra > m_inter + 0.15:
        print("\n✅ EXCELENTE: El sistema separa los temas con alta precisión.")
    elif m_intra > m_inter:
        print("\n⚠️ ACEPTABLE: Hay separación, pero los temas podrían solaparse.")
    else:
        print("\n❌ CRÍTICO: El sistema no distingue entre documentos.")

if __name__ == "__main__":
    main()
