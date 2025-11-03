from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import logging

app = Flask(__name__)
CORS(app)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Semantic Scholar
try:
    from semanticscholar import SemanticScholar
    sch = SemanticScholar()
except ImportError:
    logger.error("Please install semanticscholar: pip install semanticscholar")
    sch = None

def search_papers_by_author(author_name, limit=10):
    """Tìm papers của một tác giả"""
    try:
        authors = sch.search_author(author_name)
        if not authors or len(authors) == 0:
            logger.warning(f"No author found: {author_name}")
            return []
        
        # Lấy author đầu tiên
        author_obj = authors[0]
        papers = author_obj.papers if hasattr(author_obj, 'papers') else []
        
        result = []
        for paper in papers[:limit]:
            if paper.paperId and paper.title:
                result.append({
                    'paperId': paper.paperId,
                    'title': paper.title,
                    'year': paper.year if hasattr(paper, 'year') else None,
                    'citationCount': paper.citationCount if hasattr(paper, 'citationCount') else 0,
                    'authors': [a.name for a in paper.authors] if hasattr(paper, 'authors') and paper.authors else []
                })
        return result
    except Exception as e:
        logger.error(f"Error searching author {author_name}: {str(e)}")
        return []

def get_paper_citations(paper_id, max_citations=20):
    """Lấy danh sách citations của một paper"""
    try:
        paper = sch.get_paper(paper_id)
        citations = paper.citations[:max_citations] if hasattr(paper, 'citations') and paper.citations else []
        
        result = []
        for cited_paper in citations:
            if hasattr(cited_paper, 'paperId') and cited_paper.paperId and hasattr(cited_paper, 'title') and cited_paper.title:
                result.append({
                    'paperId': cited_paper.paperId,
                    'title': cited_paper.title,
                    'year': cited_paper.year if hasattr(cited_paper, 'year') else None,
                    'citationCount': cited_paper.citationCount if hasattr(cited_paper, 'citationCount') else 0,
                    'authors': [a.name for a in cited_paper.authors] if hasattr(cited_paper, 'authors') and cited_paper.authors else []
                })
        return result
    except Exception as e:
        logger.error(f"Error getting citations for {paper_id}: {str(e)}")
        return []

def build_citation_network(author_names, max_papers_per_author=10):
    """Xây dựng citation network từ danh sách tác giả"""
    all_papers = {}
    citation_graph = {}
    
    # Thu thập papers của từng author
    for author_name in author_names:
        logger.info(f"Searching papers for author: {author_name}")
        papers = search_papers_by_author(author_name, limit=max_papers_per_author)
        
        for paper in papers:
            paper_id = paper['paperId']
            if paper_id not in all_papers:
                all_papers[paper_id] = paper
                citation_graph[paper_id] = []
    
    # Thu thập citations cho mỗi paper
    for paper_id in list(all_papers.keys())[:50]:  # Limit to 50 papers to avoid timeout
        logger.info(f"Getting citations for paper: {all_papers[paper_id]['title'][:50]}...")
        citations = get_paper_citations(paper_id, max_citations=20)
        
        for cited_paper in citations:
            cited_id = cited_paper['paperId']
            # Thêm cited paper vào graph nếu chưa có
            if cited_id not in all_papers:
                all_papers[cited_id] = cited_paper
                citation_graph[cited_id] = []
            
            # Thêm edge: cited_paper -> paper (vì cited_paper trích dẫn paper)
            if paper_id not in citation_graph[cited_id]:
                citation_graph[cited_id].append(paper_id)
    
    return all_papers, citation_graph

def calculate_pagerank(papers, citation_graph, damping_factor=0.85, max_iterations=100):
    """Tính PageRank cho citation network"""
    if not papers:
        return []
    
    n = len(papers)
    paper_ids = list(papers.keys())
    paper_index = {pid: i for i, pid in enumerate(paper_ids)}
    
    # Initialize PageRank scores
    pagerank = np.ones(n) / n
    
    # Build adjacency matrix
    adjacency_matrix = np.zeros((n, n))
    out_degree = np.zeros(n)
    
    for source_id, targets in citation_graph.items():
        if source_id in paper_index:
            source_idx = paper_index[source_id]
            out_degree[source_idx] = len(targets)
            for target_id in targets:
                if target_id in paper_index:
                    target_idx = paper_index[target_id]
                    adjacency_matrix[source_idx][target_idx] = 1
    
    # PageRank iterations
    for iteration in range(max_iterations):
        new_pagerank = np.ones(n) * (1 - damping_factor) / n
        
        for i in range(n):
            for j in range(n):
                if adjacency_matrix[j][i] == 1 and out_degree[j] > 0:
                    new_pagerank[i] += damping_factor * pagerank[j] / out_degree[j]
        
        # Normalize
        pagerank_sum = np.sum(new_pagerank)
        if pagerank_sum > 0:
            new_pagerank = new_pagerank / pagerank_sum
        
        # Check convergence
        if np.linalg.norm(new_pagerank - pagerank) < 1e-6:
            logger.info(f"PageRank converged after {iteration + 1} iterations")
            break
        
        pagerank = new_pagerank
    
    # Create results
    results = []
    for i, paper_id in enumerate(paper_ids):
        paper = papers[paper_id]
        results.append({
            'paperId': paper_id,
            'title': paper['title'],
            'authors': paper['authors'],
            'year': paper['year'],
            'citationCount': paper['citationCount'],
            'pagerank': float(pagerank[i])
        })
    
    # Sort by PageRank score
    results.sort(key=lambda x: x['pagerank'], reverse=True)
    
    return results

def calculate_network_metrics_simple(papers, citation_graph, results):
    """Tính các network metrics cơ bản cho citation network"""
    n = len(papers)
    if n == 0:
        return {}
    
    paper_ids = list(papers.keys())
    
    # Calculate in-degree and out-degree
    in_degree = [0] * n
    out_degree = [0] * n
    
    paper_index = {pid: i for i, pid in enumerate(paper_ids)}
    
    for source_id, targets in citation_graph.items():
        if source_id in paper_index:
            source_idx = paper_index[source_id]
            out_degree[source_idx] = len(targets)
            for target_id in targets:
                if target_id in paper_index:
                    target_idx = paper_index[target_id]
                    in_degree[target_idx] += 1
    
    # Network density
    total_possible_edges = n * (n - 1)
    total_edges = sum(len(targets) for targets in citation_graph.values())
    density = total_edges / total_possible_edges if total_possible_edges > 0 else 0
    
    # Average degrees
    avg_in_degree = sum(in_degree) / n if n > 0 else 0
    avg_out_degree = sum(out_degree) / n if n > 0 else 0
    
    # Strongly connected, dangling, isolated nodes
    strongly_connected = sum(1 for i in range(n) if in_degree[i] > 0 and out_degree[i] > 0)
    dangling_nodes = sum(1 for i in range(n) if out_degree[i] == 0)
    isolated_nodes = sum(1 for i in range(n) if in_degree[i] == 0 and out_degree[i] == 0)
    
    # Simple clustering coefficient estimate
    avg_clustering = 0.5 if density > 0.3 else 0.2
    
    # Top authorities (high in-degree)
    authorities = []
    sorted_by_indegree = sorted(enumerate(in_degree), key=lambda x: x[1], reverse=True)
    for idx, deg in sorted_by_indegree[:5]:
        if deg > 0:
            paper_id = paper_ids[idx]
            authorities.append({
                'url': papers[paper_id]['title'],
                'in_degree': deg,
                'score': deg / max(in_degree) if max(in_degree) > 0 else 0
            })
    
    # Top hubs (high out-degree)
    hubs = []
    sorted_by_outdegree = sorted(enumerate(out_degree), key=lambda x: x[1], reverse=True)
    for idx, deg in sorted_by_outdegree[:5]:
        if deg > 0:
            paper_id = paper_ids[idx]
            hubs.append({
                'url': papers[paper_id]['title'],
                'out_degree': deg,
                'score': deg / max(out_degree) if max(out_degree) > 0 else 0
            })
    
    # Hub and authority scores (simplified HITS)
    hub_scores = [float(od) / max(out_degree) if max(out_degree) > 0 else 0 for od in out_degree]
    authority_scores = [float(id) / max(in_degree) if max(in_degree) > 0 else 0 for id in in_degree]
    
    return {
        'total_nodes': n,
        'total_edges': total_edges,
        'density': round(density, 4),
        'avg_in_degree': round(avg_in_degree, 2),
        'avg_out_degree': round(avg_out_degree, 2),
        'in_degree': in_degree,
        'out_degree': out_degree,
        'strongly_connected_nodes': strongly_connected,
        'dangling_nodes': dangling_nodes,
        'isolated_nodes': isolated_nodes,
        'avg_clustering_coefficient': avg_clustering,
        'hubs': hubs,
        'authorities': authorities,
        'hub_scores': hub_scores[:50],
        'authority_scores': authority_scores[:50],
        'degree_distribution': {}
    }

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
    <head>
        <title>Citation Network PageRank API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            code { background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; }
            pre { background-color: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }
            h3 { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Citation Network PageRank API</h1>
        <p>This API analyzes citation networks in academic research using PageRank algorithm.</p>
        
        <h2>Available Endpoints:</h2>
        
        <h3>1. Calculate Citation PageRank</h3>
        <code>POST /api/calculate-citation-pagerank</code>
        <p>Analyzes citation networks for given authors.</p>
        <p>Example request:</p>
        <pre>
{
  "authors": ["Geoffrey Hinton", "Yoshua Bengio"],
  "damping_factor": 0.85,
  "max_iterations": 100
}
        </pre>
        
        <h3>2. Search Author Papers</h3>
        <code>POST /api/search-author</code>
        <p>Retrieves papers by a specific author.</p>
        <p>Example request:</p>
        <pre>
{
  "authorName": "Geoffrey Hinton"
}
        </pre>
    </body>
    </html>
    """

@app.route('/api/calculate-citation-pagerank', methods=['POST'])
def calculate_citation_pagerank():
    """API endpoint để tính PageRank cho citation network"""
    try:
        data = request.get_json()
        author_names = data.get('authors', [])
        damping_factor = data.get('damping_factor', 0.85)
        max_iterations = data.get('max_iterations', 100)
        
        if not author_names or len(author_names) == 0:
            return jsonify({'error': 'Please provide at least one author name'}), 400
        
        if not sch:
            return jsonify({'error': 'Semantic Scholar API not available. Please install: pip install semanticscholar'}), 500
        
        logger.info(f"Calculating PageRank for authors: {author_names}")
        
        # Build citation network
        papers, citation_graph = build_citation_network(author_names)
        
        if len(papers) == 0:
            return jsonify({'error': 'No papers found for the given authors'}), 404
        
        # Calculate PageRank
        results = calculate_pagerank(papers, citation_graph, damping_factor, max_iterations)
        
        # Prepare network data for visualization
        nodes = []
        edges = []
        
        for paper_id, paper in papers.items():
            nodes.append({
                'id': paper_id,
                'label': paper['title'][:50] + '...' if len(paper['title']) > 50 else paper['title'],
                'citationCount': paper['citationCount']
            })
        
        for source_id, targets in citation_graph.items():
            for target_id in targets:
                edges.append({
                    'from': source_id,
                    'to': target_id
                })
        
        network_metrics = calculate_network_metrics_simple(papers, citation_graph, results)
        return jsonify({
            'results': results[:50],  # Top 50 papers
            'network': {
                'nodes': nodes,
                'edges': edges
            },
            'stats': {
                'totalPapers': len(papers),
                'totalCitations': sum(len(targets) for targets in citation_graph.values())
            },
            'networkMetrics': network_metrics
        })
        
    except Exception as e:
        logger.error(f"Error calculating PageRank: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search-author', methods=['POST'])
def search_author():
    """API endpoint để search author và lấy papers"""
    try:
        data = request.get_json()
        author_name = data.get('authorName', '')
        
        if not author_name:
            return jsonify({'error': 'Please provide an author name'}), 400
        
        if not sch:
            return jsonify({'error': 'Semantic Scholar API not available'}), 500
        
        papers = search_papers_by_author(author_name, limit=20)
        
        return jsonify({
            'author': author_name,
            'papers': papers,
            'count': len(papers)
        })
        
    except Exception as e:
        logger.error(f"Error searching author: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)