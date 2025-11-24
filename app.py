from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import csv
from io import StringIO
from datetime import datetime
import numpy as np
import logging
import time
import os 
from dotenv import load_dotenv
from flask import Response, stream_with_context

load_dotenv()

app = Flask(__name__)
CORS(app)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= PHÂN QUYỀN USER ROLES =============
USER_ROLES = {
    'researcher': {
        'name': 'Researcher/University/Institute',
        'description': 'For academic researchers and institutions',
        'permissions': {
            'view_pagerank': True,
            'view_network_graph': True,
            'view_basic_stats': True,
            'view_network_metrics': False,
            'export_data': False,
            'advanced_analysis': False,
            'customize_parameters': False 
        }
    },
    'data_scientist': {
        'name': 'Data Scientist',
        'description': 'For technical users and researchers',
        'permissions': {
            'view_pagerank': True,
            'view_network_graph': True,
            'view_basic_stats': True,
            'view_network_metrics': True,
            'export_data': True,
            'advanced_analysis': True,
            'customize_parameters': True  
        }
    }
}

# Rate limiting config
API_DELAY = 1.1
MAX_PAPERS_TO_PROCESS = 50
MAX_CITATIONS_PER_PAPER = 20

# Import Semantic Scholar
try:
    from semanticscholar import SemanticScholar
    
    API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY')  
    
    if API_KEY:
        sch = SemanticScholar(api_key=API_KEY)
        logger.info(f"✅ Using Semantic Scholar API with key: {API_KEY[:6]}***")
        logger.info("📊 Rate limit: 1 request/second")
    else:
        sch = SemanticScholar()
        logger.warning("⚠️ No API key found in .env file!")
        logger.warning("⚠️ Using rate-limited anonymous access")
        
except ImportError:
    logger.error("❌ Please install required packages:")
    logger.error("   pip install semanticscholar python-dotenv")
    sch = None

# ============= HELPER FUNCTIONS =============
def search_papers_by_author(author_name, limit=10):
    """Tìm papers của một tác giả"""
    try:
        time.sleep(API_DELAY)
        authors = sch.search_author(author_name)
        if not authors or len(authors) == 0:
            logger.warning(f"No author found: {author_name}")
            return []
        
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
        time.sleep(2)
        return []

def get_paper_citations(paper_id, max_citations=20):
    """Lấy danh sách citations của một paper"""
    try:
        time.sleep(API_DELAY)
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
        time.sleep(2)
        return []

def build_citation_network(author_names, max_papers_per_author=10):
    """Xây dựng citation network từ danh sách tác giả"""
    all_papers = {}
    citation_graph = {}
    
    logger.info(f"👤 Fetching papers from {len(author_names)} authors...")
    for idx, author_name in enumerate(author_names, 1):
        logger.info(f"[{idx}/{len(author_names)}] Searching: {author_name}")
        papers = search_papers_by_author(author_name, limit=max_papers_per_author)
        
        logger.info(f"   ✅ Found {len(papers)} papers")
        for paper in papers:
            paper_id = paper['paperId']
            if paper_id not in all_papers:
                all_papers[paper_id] = paper
                citation_graph[paper_id] = []
    
    paper_ids_to_process = list(all_papers.keys())[:MAX_PAPERS_TO_PROCESS]
    logger.info(f"🔗 Fetching citations for {len(paper_ids_to_process)} papers...")
    logger.info(f"⏱️  Estimated time: ~{len(paper_ids_to_process) * 1.2:.0f} seconds")
    
    for idx, paper_id in enumerate(paper_ids_to_process, 1):
        logger.info(f"[{idx}/{len(paper_ids_to_process)}] Citations: {all_papers[paper_id]['title'][:50]}...")
        citations = get_paper_citations(paper_id, max_citations=MAX_CITATIONS_PER_PAPER)
        
        logger.info(f"   ✅ {len(citations)} citations")
        for cited_paper in citations:
            cited_id = cited_paper['paperId']
            if cited_id not in all_papers:
                all_papers[cited_id] = cited_paper
                citation_graph[cited_id] = []
            
            if paper_id not in citation_graph[cited_id]:
                citation_graph[cited_id].append(paper_id)
    
    logger.info(f"✅ Network complete: {len(all_papers)} papers, {sum(len(v) for v in citation_graph.values())} citations")
    return all_papers, citation_graph

def search_paper_by_title(title, limit=1):
    """Tìm paper theo tiêu đề"""
    try:
        time.sleep(API_DELAY)
        papers = sch.search_paper(title, limit=limit)
        if not papers or len(papers) == 0:
            logger.warning(f"No paper found: {title}")
            return []
        
        result = []
        for paper in papers:
            if hasattr(paper, 'paperId') and paper.paperId and hasattr(paper, 'title') and paper.title:
                result.append({
                    'paperId': paper.paperId,
                    'title': paper.title,
                    'year': paper.year if hasattr(paper, 'year') else None,
                    'citationCount': paper.citationCount if hasattr(paper, 'citationCount') else 0,
                    'authors': [a.name for a in paper.authors] if hasattr(paper, 'authors') and paper.authors else []
                })
        return result
    except Exception as e:
        logger.error(f"Error searching paper {title}: {str(e)}")
        time.sleep(2)
        return []
    
def get_paper_by_doi(identifier):
    """Lấy paper từ DOI hoặc arXiv ID"""
    try:
        time.sleep(API_DELAY)
        
        paper = None
        
        if 'arxiv' in identifier.lower():
            arxiv_id = identifier.split('arXiv.')[-1]
            logger.info(f"   Detected arXiv DOI, extracting ID: {arxiv_id}")
            
            try:
                paper = sch.get_paper(f"arXiv:{arxiv_id}")
                logger.info(f"   ✅ Found via arXiv ID: arXiv:{arxiv_id}")
            except:
                try:
                    paper = sch.get_paper(arxiv_id)
                    logger.info(f"   ✅ Found via plain arXiv ID: {arxiv_id}")
                except:
                    logger.warning(f"   ⚠️ arXiv lookup failed, trying DOI...")
        
        elif identifier.replace('.', '').isdigit() and '.' in identifier:
            logger.info(f"   Detected plain arXiv ID: {identifier}")
            try:
                paper = sch.get_paper(f"arXiv:{identifier}")
                logger.info(f"   ✅ Found via arXiv ID: arXiv:{identifier}")
            except:
                try:
                    paper = sch.get_paper(identifier)
                    logger.info(f"   ✅ Found via plain ID: {identifier}")
                except:
                    pass
        
        if paper is None:
            try:
                paper = sch.get_paper(f"DOI:{identifier}")
                logger.info(f"   ✅ Found via DOI: {identifier}")
            except:
                try:
                    paper = sch.get_paper(identifier)
                    logger.info(f"   ✅ Found via direct lookup: {identifier}")
                except:
                    pass
        
        if paper and hasattr(paper, 'paperId') and paper.paperId:
            return {
                'paperId': paper.paperId,
                'title': paper.title if hasattr(paper, 'title') else 'Unknown',
                'year': paper.year if hasattr(paper, 'year') else None,
                'citationCount': paper.citationCount if hasattr(paper, 'citationCount') else 0,
                'authors': [a.name for a in paper.authors] if hasattr(paper, 'authors') and paper.authors else []
            }
        else:
            logger.warning(f"   ❌ Paper not found for identifier: {identifier}")
            return None
            
    except Exception as e:
        logger.error(f"   ❌ Error getting paper by identifier {identifier}: {str(e)}")
        time.sleep(2)
        return None

def build_citation_network_from_papers(paper_identifiers, max_citations=20, input_type='doi'):
    """Xây dựng citation network từ danh sách papers"""
    all_papers = {}
    citation_graph = {}
    
    logger.info(f"📚 Searching for {len(paper_identifiers)} papers by {input_type.upper()}...")
    
    for idx, identifier in enumerate(paper_identifiers, 1):
        if input_type == 'doi':
            logger.info(f"[{idx}/{len(paper_identifiers)}] Looking up DOI: {identifier}")
            paper = get_paper_by_doi(identifier)
            
            if paper:
                paper_id = paper['paperId']
                if paper_id not in all_papers:
                    all_papers[paper_id] = paper
                    citation_graph[paper_id] = []
                    logger.info(f"   ✅ {paper['title'][:60]}")
            else:
                logger.warning(f"   ❌ Paper not found for DOI: {identifier}")
        else:
            logger.info(f"[{idx}/{len(paper_identifiers)}] Searching title: {identifier[:60]}...")
            papers = search_paper_by_title(identifier, limit=1)
            
            if papers and len(papers) > 0:
                paper = papers[0]
                paper_id = paper['paperId']
                if paper_id not in all_papers:
                    all_papers[paper_id] = paper
                    citation_graph[paper_id] = []
                    logger.info(f"   ✅ {paper['title'][:60]}")
            else:
                logger.warning(f"   ❌ Not found: {identifier[:60]}")
    
    if len(all_papers) == 0:
        logger.error("❌ No papers found!")
        return all_papers, citation_graph
    
    paper_ids_to_process = list(all_papers.keys())[:MAX_PAPERS_TO_PROCESS]
    logger.info(f"🔗 Fetching citations for {len(paper_ids_to_process)} papers...")
    logger.info(f"⏱️  Estimated time: ~{len(paper_ids_to_process) * 1.2:.0f} seconds")
    
    for idx, paper_id in enumerate(paper_ids_to_process, 1):
        logger.info(f"[{idx}/{len(paper_ids_to_process)}] Citations: {all_papers[paper_id]['title'][:50]}...")
        citations = get_paper_citations(paper_id, max_citations=max_citations)
        
        logger.info(f"   ✅ {len(citations)} citations")
        for cited_paper in citations:
            cited_id = cited_paper['paperId']
            if cited_id not in all_papers:
                all_papers[cited_id] = cited_paper
                citation_graph[cited_id] = []
            
            if paper_id not in citation_graph[cited_id]:
                citation_graph[cited_id].append(paper_id)
    
    logger.info(f"✅ Network complete: {len(all_papers)} papers, {sum(len(v) for v in citation_graph.values())} citations")
    return all_papers, citation_graph


def calculate_pagerank(papers, citation_graph, damping_factor=0.85, max_iterations=100):
    """Tính PageRank cho citation network"""
    if not papers:
        return [], []
    
    n = len(papers)
    paper_ids = list(papers.keys())
    paper_index = {pid: i for i, pid in enumerate(paper_ids)}
    
    pagerank = np.ones(n) / n
    
    # Track residuals for convergence curve
    residuals = []
    
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
    
    for iteration in range(max_iterations):
        new_pagerank = np.ones(n) * (1 - damping_factor) / n
        
        for i in range(n):
            for j in range(n):
                if adjacency_matrix[j][i] == 1 and out_degree[j] > 0:
                    new_pagerank[i] += damping_factor * pagerank[j] / out_degree[j]
        
        pagerank_sum = np.sum(new_pagerank)
        if pagerank_sum > 0:
            new_pagerank = new_pagerank / pagerank_sum
        
        # Calculate residual
        residual = np.linalg.norm(new_pagerank - pagerank)
        residuals.append(float(residual))
        
        if residual < 1e-6:
            logger.info(f"PageRank converged after {iteration + 1} iterations")
            break
        
        pagerank = new_pagerank
    
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
    
    results.sort(key=lambda x: x['pagerank'], reverse=True)
    
    return results, residuals


def calculate_hits(papers, citation_graph, max_iterations=100, convergence_threshold=1e-6):
    """Tính HITS với convergence tracking"""
    if not papers:
        return [], [], 0, []
    
    n = len(papers)
    paper_ids = list(papers.keys())
    paper_index = {pid: i for i, pid in enumerate(paper_ids)}
    
    authority = np.ones(n)
    hub = np.ones(n)
    auth_residuals = []
    hub_residuals = []
    
    adjacency_matrix = np.zeros((n, n))
    
    for source_id, targets in citation_graph.items():
        if source_id in paper_index:
            source_idx = paper_index[source_id]
            for target_id in targets:
                if target_id in paper_index:
                    target_idx = paper_index[target_id]
                    adjacency_matrix[source_idx][target_idx] = 1
    
    for iteration in range(max_iterations):
        new_authority = adjacency_matrix.T @ hub
        new_hub = adjacency_matrix @ new_authority
        
        auth_norm = np.linalg.norm(new_authority)
        hub_norm = np.linalg.norm(new_hub)
        
        if auth_norm > 0:
            new_authority = new_authority / auth_norm
        if hub_norm > 0:
            new_hub = new_hub / hub_norm
        
        # Track residual (combined authority + hub changes)
        auth_diff = np.linalg.norm(new_authority - authority)
        hub_diff = np.linalg.norm(new_hub - hub)
        auth_residuals.append(float(auth_diff))
        hub_residuals.append(float(hub_diff))
        
        if auth_diff < convergence_threshold and hub_diff < convergence_threshold:
            logger.info(f"HITS converged after {iteration + 1} iterations")
            break
        
        authority = new_authority
        hub = new_hub
    
    authority_results = []
    hub_results = []
    
    for i, paper_id in enumerate(paper_ids):
        paper = papers[paper_id]
        base_result = {
            'paperId': paper_id,
            'title': paper['title'],
            'authors': paper['authors'],
            'year': paper['year'],
            'citationCount': paper['citationCount']
        }
        
        authority_results.append({**base_result, 'authority_score': float(authority[i])})
        hub_results.append({**base_result, 'hub_score': float(hub[i])})
    
    authority_results.sort(key=lambda x: x['authority_score'], reverse=True)
    hub_results.sort(key=lambda x: x['hub_score'], reverse=True)

    combined_residuals = [(a + h) / 2 for a, h in zip(auth_residuals, hub_residuals)]

    return authority_results, hub_results, iteration + 1, combined_residuals


def calculate_weighted_pagerank(papers, citation_graph, damping_factor=0.85, max_iterations=100):
    """Tính Weighted PageRank với convergence tracking"""
    if not papers:
        return [], [], []
    
    n = len(papers)
    paper_ids = list(papers.keys())
    paper_index = {pid: i for i, pid in enumerate(paper_ids)}
    
    pagerank = np.ones(n) / n
    residuals = []
    
    # Build weighted adjacency matrix
    weighted_adjacency = np.zeros((n, n))
    out_weight = np.zeros(n)
    
    citation_counts = np.array([papers[pid]['citationCount'] for pid in paper_ids])
    max_citations = max(citation_counts) if max(citation_counts) > 0 else 1
    
    for source_id, targets in citation_graph.items():
        if source_id in paper_index:
            source_idx = paper_index[source_id]
            
            for target_id in targets:
                if target_id in paper_index:
                    target_idx = paper_index[target_id]
                    target_citations = papers[target_id]['citationCount']
                    weight = 1 + (target_citations / max_citations)
                    
                    weighted_adjacency[source_idx][target_idx] = weight
                    out_weight[source_idx] += weight
    
    for iteration in range(max_iterations):
        new_pagerank = np.ones(n) * (1 - damping_factor) / n
        
        for i in range(n):
            for j in range(n):
                if weighted_adjacency[j][i] > 0 and out_weight[j] > 0:
                    new_pagerank[i] += damping_factor * pagerank[j] * (weighted_adjacency[j][i] / out_weight[j])
        
        pagerank_sum = np.sum(new_pagerank)
        if pagerank_sum > 0:
            new_pagerank = new_pagerank / pagerank_sum
        
        # track residual
        residual = np.linalg.norm(new_pagerank - pagerank)
        residuals.append(float(residual))
        
        if residual < 1e-6:
            logger.info(f"Weighted PageRank converged after {iteration + 1} iterations")
            break
        
        pagerank = new_pagerank
    
    results = []
    for i, paper_id in enumerate(paper_ids):
        paper = papers[paper_id]
        results.append({
            'paperId': paper_id,
            'title': paper['title'],
            'authors': paper['authors'],
            'year': paper['year'],
            'citationCount': paper['citationCount'],
            'weighted_pagerank': float(pagerank[i])
        })
    
    results.sort(key=lambda x: x['weighted_pagerank'], reverse=True)
    
    return results, iteration + 1, residuals


def calculate_correlation(results1, results2, score_key1='pagerank', score_key2='authority_score'):
    """
    Tính Spearman rank correlation giữa 2 algorithms
    """
    from scipy.stats import spearmanr
    
    # Extract common papers
    ids1 = {r['paperId']: r[score_key1] for r in results1}
    ids2 = {r['paperId']: r[score_key2] for r in results2}
    
    common_ids = set(ids1.keys()) & set(ids2.keys())
    
    if len(common_ids) < 2:
        return 0.0
    
    scores1 = [ids1[pid] for pid in common_ids]
    scores2 = [ids2[pid] for pid in common_ids]
    
    correlation, _ = spearmanr(scores1, scores2)
    
    return float(correlation)


def calculate_top_k_overlap(results1, results2, k=10):
    """
    Tính overlap giữa top-k papers của 2 algorithms
    """
    top_k_ids1 = {r['paperId'] for r in results1[:k]}
    top_k_ids2 = {r['paperId'] for r in results2[:k]}
    
    overlap = len(top_k_ids1 & top_k_ids2)
    
    return overlap / k if k > 0 else 0.0



def calculate_network_metrics_simple(papers, citation_graph, results):
    """Tính các network metrics cơ bản cho citation network"""
    n = len(papers)
    if n == 0:
        return {}
    
    paper_ids = list(papers.keys())
    
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
    
    total_possible_edges = n * (n - 1)
    total_edges = sum(len(targets) for targets in citation_graph.values())
    density = total_edges / total_possible_edges if total_possible_edges > 0 else 0
    
    avg_in_degree = sum(in_degree) / n if n > 0 else 0
    avg_out_degree = sum(out_degree) / n if n > 0 else 0
    
    strongly_connected = sum(1 for i in range(n) if in_degree[i] > 0 and out_degree[i] > 0)
    dangling_nodes = sum(1 for i in range(n) if out_degree[i] == 0)
    isolated_nodes = sum(1 for i in range(n) if in_degree[i] == 0 and out_degree[i] == 0)
    
    avg_clustering = 0.5 if density > 0.3 else 0.2
    
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

def send_progress(status, progress, message, **extra_data):
    """Helper function to send progress updates via SSE"""
    data = {
        'status': status,
        'progress': progress,
        'message': message,
        **extra_data
    }
    return f"data: {json.dumps(data)}\n\n"

def build_citation_network_with_progress(author_names, max_papers_per_author=10):
    """Xây dựng citation network với progress tracking"""
    all_papers = {}
    citation_graph = {}
    
    total_authors = len(author_names)
    yield send_progress('fetching_authors', 5, f'📚 Fetching papers from {total_authors} authors...')
    
    logger.info(f"👤 Fetching papers from {total_authors} authors...")
    for idx, author_name in enumerate(author_names, 1):
        progress = 5 + int((idx / total_authors) * 25)  # 5% -> 30%
        yield send_progress(
            'fetching_author', 
            progress, 
            f'[{idx}/{total_authors}] 🔍 Searching: {author_name}',
            current_author=author_name,
            current_step=idx,
            total_steps=total_authors
        )
        
        logger.info(f"[{idx}/{total_authors}] Searching: {author_name}")
        papers = search_papers_by_author(author_name, limit=max_papers_per_author)
        
        logger.info(f"   ✅ Found {len(papers)} papers")
        for paper in papers:
            paper_id = paper['paperId']
            if paper_id not in all_papers:
                all_papers[paper_id] = paper
                citation_graph[paper_id] = []
        
        yield send_progress(
            'author_complete', 
            progress, 
            f'✅ Found {len(papers)} papers from {author_name}'
        )
    
    # Fetch citations
    paper_ids_to_process = list(all_papers.keys())[:MAX_PAPERS_TO_PROCESS]
    total_papers = len(paper_ids_to_process)
    
    yield send_progress(
        'fetching_citations', 
        30, 
        f'🔗 Fetching citations for {total_papers} papers...',
        total_papers=total_papers
    )
    
    logger.info(f"🔗 Fetching citations for {total_papers} papers...")
    logger.info(f"⏱️  Estimated time: ~{total_papers * 1.2:.0f} seconds")
    
    for idx, paper_id in enumerate(paper_ids_to_process, 1):
        progress = 30 + int((idx / total_papers) * 50)  # 30% -> 80%
        paper_title = all_papers[paper_id]['title'][:50]
        
        yield send_progress(
            'fetching_citation',
            progress,
            f'[{idx}/{total_papers}] 📄 {paper_title}...',
            current_paper=idx,
            total_papers=total_papers
        )
        
        logger.info(f"[{idx}/{total_papers}] Citations: {paper_title}...")
        citations = get_paper_citations(paper_id, max_citations=MAX_CITATIONS_PER_PAPER)
        
        logger.info(f"   ✅ {len(citations)} citations")
        for cited_paper in citations:
            cited_id = cited_paper['paperId']
            if cited_id not in all_papers:
                all_papers[cited_id] = cited_paper
                citation_graph[cited_id] = []
            
            if paper_id not in citation_graph[cited_id]:
                citation_graph[cited_id].append(paper_id)
    
    logger.info(f"✅ Network complete: {len(all_papers)} papers, {sum(len(v) for v in citation_graph.values())} citations")
    
    yield send_progress('network_complete', 80, f'✅ Network built: {len(all_papers)} papers')
    yield (all_papers, citation_graph)


def build_citation_network_from_papers_with_progress(paper_identifiers, max_citations=20, input_type='doi'):
    """Xây dựng citation network từ papers với progress tracking"""
    all_papers = {}
    citation_graph = {}
    
    total_identifiers = len(paper_identifiers)
    yield send_progress('searching_papers', 5, f'📚 Searching for {total_identifiers} papers by {input_type.upper()}...')
    
    logger.info(f"📚 Searching for {total_identifiers} papers by {input_type.upper()}...")
    
    for idx, identifier in enumerate(paper_identifiers, 1):
        progress = 5 + int((idx / total_identifiers) * 25)  # 5% -> 30%
        
        if input_type == 'doi':
            yield send_progress('looking_up_doi', progress, f'[{idx}/{total_identifiers}] 🔍 Looking up DOI: {identifier[:40]}...')
            logger.info(f"[{idx}/{total_identifiers}] Looking up DOI: {identifier}")
            paper = get_paper_by_doi(identifier)
            
            if paper:
                paper_id = paper['paperId']
                if paper_id not in all_papers:
                    all_papers[paper_id] = paper
                    citation_graph[paper_id] = []
                    yield send_progress('paper_found', progress, f'✅ {paper["title"][:60]}')
                    logger.info(f"   ✅ {paper['title'][:60]}")
            else:
                yield send_progress('paper_not_found', progress, f'❌ Paper not found for DOI: {identifier[:40]}')
                logger.warning(f"   ❌ Paper not found for DOI: {identifier}")
        else:
            yield send_progress('searching_title', progress, f'[{idx}/{total_identifiers}] 🔍 Searching: {identifier[:60]}...')
            logger.info(f"[{idx}/{total_identifiers}] Searching title: {identifier[:60]}...")
            papers = search_paper_by_title(identifier, limit=1)
            
            if papers and len(papers) > 0:
                paper = papers[0]
                paper_id = paper['paperId']
                if paper_id not in all_papers:
                    all_papers[paper_id] = paper
                    citation_graph[paper_id] = []
                    yield send_progress('paper_found', progress, f'✅ {paper["title"][:60]}')
                    logger.info(f"   ✅ {paper['title'][:60]}")
            else:
                yield send_progress('paper_not_found', progress, f'❌ Not found: {identifier[:60]}')
                logger.warning(f"   ❌ Not found: {identifier[:60]}")
    
    if len(all_papers) == 0:
        logger.error("❌ No papers found!")
        yield send_progress('error', 0, '❌ No papers found!')
        return
    
    # Fetch citations
    paper_ids_to_process = list(all_papers.keys())[:MAX_PAPERS_TO_PROCESS]
    total_papers = len(paper_ids_to_process)
    
    yield send_progress('fetching_citations', 30, f'🔗 Fetching citations for {total_papers} papers...')
    logger.info(f"🔗 Fetching citations for {total_papers} papers...")
    logger.info(f"⏱️  Estimated time: ~{total_papers * 1.2:.0f} seconds")
    
    for idx, paper_id in enumerate(paper_ids_to_process, 1):
        progress = 30 + int((idx / total_papers) * 50)  # 30% -> 80%
        paper_title = all_papers[paper_id]['title'][:50]
        
        yield send_progress(
            'fetching_citation',
            progress,
            f'[{idx}/{total_papers}] 📄 {paper_title}...',
            current_paper=idx,
            total_papers=total_papers
        )
        
        logger.info(f"[{idx}/{total_papers}] Citations: {paper_title}...")
        citations = get_paper_citations(paper_id, max_citations=max_citations)
        
        logger.info(f"   ✅ {len(citations)} citations")
        for cited_paper in citations:
            cited_id = cited_paper['paperId']
            if cited_id not in all_papers:
                all_papers[cited_id] = cited_paper
                citation_graph[cited_id] = []
            
            if paper_id not in citation_graph[cited_id]:
                citation_graph[cited_id].append(paper_id)
    
    logger.info(f"✅ Network complete: {len(all_papers)} papers, {sum(len(v) for v in citation_graph.values())} citations")
    yield send_progress('network_complete', 80, f'✅ Network built: {len(all_papers)} papers')
    yield (all_papers, citation_graph)

# ============= API ENDPOINTS =============
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
        
        <h3>2. Get User Roles</h3>
        <code>GET /api/roles</code>
        <p>Returns available user roles and their permissions.</p>
    </body>
    </html>
    """

# ============= MỚI: API LẤY DANH SÁCH ROLES =============
@app.route('/api/roles', methods=['GET'])
def get_roles():
    """API endpoint để lấy danh sách roles và permissions"""
    return jsonify(USER_ROLES)

@app.route('/api/calculate-citation-pagerank', methods=['POST'])
def calculate_citation_pagerank():
    try:
        data = request.get_json()
        author_names = data.get('authors', [])
        damping_factor = data.get('damping_factor', 0.85)
        max_iterations = data.get('max_iterations', 100)
        user_role = data.get('user_role', 'researcher')
        
        if not author_names or len(author_names) == 0:
            return jsonify({'error': 'Please provide at least one author name'}), 400
        
        if not sch:
            return jsonify({'error': 'Semantic Scholar API not available'}), 500
        
        logger.info(f"Calculating PageRank for authors: {author_names} (Role: {user_role})")
        
        # Build citation network
        papers, citation_graph = build_citation_network(author_names)
        
        if len(papers) == 0:
            return jsonify({'error': 'No papers found'}), 404
        
        # Calculate PageRank
        results = calculate_pagerank(papers, citation_graph, damping_factor, max_iterations)
        
        # Prepare network data
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
        
        # LẤY PERMISSIONS THỰC TẾ
        permissions = USER_ROLES.get(user_role, USER_ROLES['researcher'])['permissions']
        
        response_data = {
            'results': results[:50],
            'network': {
                'nodes': nodes,
                'edges': edges
            },
            'stats': {
                'totalPapers': len(papers),
                'totalCitations': sum(len(targets) for targets in citation_graph.values())
            },
            'userRole': user_role,
            'permissions': permissions  # Trả về permissions thực tế
        }
        
        # CHỈ THÊM METRICS NẾU CÓ QUYỀN
        if permissions.get('view_network_metrics', False):
            network_metrics = calculate_network_metrics_simple(papers, citation_graph, results)
            response_data['networkMetrics'] = network_metrics
        
        return jsonify(response_data)
        
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

@app.route('/api/calculate-citation-pagerank-by-papers', methods=['POST'])
def calculate_citation_pagerank_by_papers():
    try:
        data = request.get_json()
        paper_identifiers = data.get('papers', [])
        input_type = data.get('input_type', 'doi')
        damping_factor = data.get('damping_factor', 0.85)
        max_iterations = data.get('max_iterations', 100)
        user_role = data.get('user_role', 'researcher')
        
        if not paper_identifiers or len(paper_identifiers) == 0:
            return jsonify({'error': f'Please provide at least one paper {input_type}'}), 400
        
        if not sch:
            return jsonify({'error': 'Semantic Scholar API not available'}), 500
        
        logger.info(f"Calculating PageRank for {len(paper_identifiers)} papers by {input_type.upper()} (Role: {user_role})")
        
        papers, citation_graph = build_citation_network_from_papers(
            paper_identifiers, 
            max_citations=MAX_CITATIONS_PER_PAPER,
            input_type=input_type
        )
        
        if len(papers) == 0:
            return jsonify({'error': f'No papers found'}), 404
        
        results = calculate_pagerank(papers, citation_graph, damping_factor, max_iterations)
        
        # Prepare network data
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
        
        # LẤY PERMISSIONS THỰC TẾ
        permissions = USER_ROLES.get(user_role, USER_ROLES['researcher'])['permissions']
        
        response_data = {
            'results': results[:50],
            'network': {
                'nodes': nodes,
                'edges': edges
            },
            'stats': {
                'totalPapers': len(papers),
                'totalCitations': sum(len(targets) for targets in citation_graph.values())
            },
            'userRole': user_role,
            'permissions': permissions
        }
        
        # CHỈ THÊM METRICS NẾU CÓ QUYỀN
        if permissions.get('view_network_metrics', False):
            network_metrics = calculate_network_metrics_simple(papers, citation_graph, results)
            response_data['networkMetrics'] = network_metrics
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error calculating PageRank: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/api/export-data', methods=['POST'])
def export_data():
    """
    API endpoint để export toàn bộ data
    Chỉ dành cho Data Scientist
    """
    try:
        data = request.get_json()
        user_role = data.get('user_role', 'researcher')
        export_format = data.get('format', 'json')  # 'json' hoặc 'csv'
        
        # Kiểm tra quyền
        permissions = USER_ROLES.get(user_role, USER_ROLES['researcher'])['permissions']
        if not permissions.get('export_data', False):
            return jsonify({'error': 'Permission denied. Only Data Scientists can export data.'}), 403
        
        # Lấy data từ request (frontend sẽ gửi toàn bộ data)
        results = data.get('results', [])
        network = data.get('network', {})
        stats = data.get('stats', {})
        network_metrics = data.get('networkMetrics', {})
        input_mode = data.get('input_mode', 'unknown')
        parameters = data.get('parameters', {})
        
        # Tạo export data structure
        export_data_obj = {
            'metadata': {
                'export_date': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'user_role': user_role,
                'input_mode': input_mode,
                'parameters': parameters,
                'total_papers': len(results),
                'export_format': export_format
            },
            'results': results,
            'network': network,
            'stats': stats,
            'networkMetrics': network_metrics
        }
        
        if export_format == 'json':
            # Export as JSON
            return jsonify({
                'success': True,
                'data': export_data_obj,
                'filename': f'citation_network_export_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.json'
            })
        
        elif export_format == 'csv':
            # Export as CSV (flatten structure)
            # Tạo CSV với results table
            csv_output = StringIO()
            
            if len(results) > 0:
                # CSV Headers
                fieldnames = ['rank', 'paperId', 'title', 'authors', 'year', 'citationCount', 'pagerank']
                writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write rows
                for idx, result in enumerate(results, 1):
                    writer.writerow({
                        'rank': idx,
                        'paperId': result.get('paperId', ''),
                        'title': result.get('title', ''),
                        'authors': '; '.join(result.get('authors', [])),
                        'year': result.get('year', ''),
                        'citationCount': result.get('citationCount', 0),
                        'pagerank': result.get('pagerank', 0)
                    })
                
                # Add metadata section
                csv_output.write('\n\n')
                csv_output.write('# METADATA\n')
                csv_output.write(f'# Export Date: {export_data_obj["metadata"]["export_date"]}\n')
                csv_output.write(f'# Total Papers: {export_data_obj["metadata"]["total_papers"]}\n')
                csv_output.write(f'# Damping Factor: {parameters.get("damping_factor", "N/A")}\n')
                csv_output.write(f'# Max Iterations: {parameters.get("max_iterations", "N/A")}\n')
                
                # Add network stats
                csv_output.write('\n# NETWORK STATISTICS\n')
                csv_output.write(f'# Total Citations: {stats.get("totalCitations", 0)}\n')
                csv_output.write(f'# Network Density: {network_metrics.get("density", 0):.4f}\n')
                csv_output.write(f'# Avg Clustering: {network_metrics.get("avg_clustering_coefficient", 0):.4f}\n')
            
            csv_content = csv_output.getvalue()
            csv_output.close()
            
            return jsonify({
                'success': True,
                'data': csv_content,
                'filename': f'citation_network_export_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
            })
        
        else:
            return jsonify({'error': 'Invalid format. Use "json" or "csv"'}), 400
            
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/calculate-with-algorithm', methods=['POST'])
def calculate_with_algorithm():
    try:
        data = request.get_json()
        algorithm = data.get('algorithm', 'pagerank')
        author_names = data.get('authors', [])
        paper_identifiers = data.get('papers', [])
        input_mode = data.get('input_mode', 'authors')
        input_type = data.get('input_type', 'doi')
        damping_factor = data.get('damping_factor', 0.85)
        max_iterations = data.get('max_iterations', 100)
        user_role = data.get('user_role', 'researcher')
        
        if not sch:
            return jsonify({'error': 'Semantic Scholar API not available'}), 500
        
        # Build network
        if input_mode == 'authors':
            if not author_names or len(author_names) == 0:
                return jsonify({'error': 'Please provide at least one author name'}), 400
            papers, citation_graph = build_citation_network(author_names)
        else:
            if not paper_identifiers or len(paper_identifiers) == 0:
                return jsonify({'error': 'Please provide at least one paper'}), 400
            papers, citation_graph = build_citation_network_from_papers(
                paper_identifiers, 
                max_citations=MAX_CITATIONS_PER_PAPER,
                input_type=input_type
            )
        
        if len(papers) == 0:
            return jsonify({'error': 'No papers found'}), 404
        
        import time
        start_time = time.time()
        
        # LẤY PERMISSIONS THỰC TẾ
        permissions = USER_ROLES.get(user_role, USER_ROLES['researcher'])['permissions']
        
        # Calculate based on algorithm
        if algorithm == 'pagerank':
            results, residuals = calculate_pagerank(papers, citation_graph, damping_factor, max_iterations)  # ✅ FIX
            iterations = len(residuals)  # FIX: use actual iterations
            computation_time = time.time() - start_time
            
            response_data = {
                'algorithm': 'pagerank',
                'results': results[:50],
                'performance': {
                    'computation_time': round(computation_time, 3),
                    'iterations': iterations,
                    'papers_analyzed': len(papers)
                }
            }
            
            # CHỈ THÊM METRICS NẾU CÓ QUYỀN
            if permissions.get('view_network_metrics', False):
                network_metrics = calculate_network_metrics_simple(papers, citation_graph, results)
                response_data['networkMetrics'] = network_metrics
            
        elif algorithm == 'hits':
            authority_results, hub_results, iterations, residuals = calculate_hits(papers, citation_graph, max_iterations)  # ✅ FIX
            computation_time = time.time() - start_time
            
            response_data = {
                'algorithm': 'hits',
                'authority_results': authority_results[:50],
                'hub_results': hub_results[:50],
                'performance': {
                    'computation_time': round(computation_time, 3),
                    'iterations': iterations,
                    'papers_analyzed': len(papers)
                }
            }
            
            # CHỈ THÊM METRICS NẾU CÓ QUYỀN
            if permissions.get('view_network_metrics', False):
                network_metrics = calculate_network_metrics_simple(papers, citation_graph, authority_results)
                response_data['networkMetrics'] = network_metrics
            
        elif algorithm == 'weighted_pagerank':
            results, iterations, residuals = calculate_weighted_pagerank(papers, citation_graph, damping_factor, max_iterations)  # ✅ FIX
            computation_time = time.time() - start_time
            
            response_data = {
                'algorithm': 'weighted_pagerank',
                'results': results[:50],
                'performance': {
                    'computation_time': round(computation_time, 3),
                    'iterations': iterations,
                    'papers_analyzed': len(papers)
                }
            }
            
            # CHỈ THÊM METRICS NẾU CÓ QUYỀN
            if permissions.get('view_network_metrics', False):
                network_metrics = calculate_network_metrics_simple(papers, citation_graph, results)
                response_data['networkMetrics'] = network_metrics
        else:
            return jsonify({'error': f'Unknown algorithm: {algorithm}'}), 400
        
        # Add network data
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
        
        response_data['network'] = {
            'nodes': nodes,
            'edges': edges
        }
        
        response_data['stats'] = {
            'totalPapers': len(papers),
            'totalCitations': sum(len(targets) for targets in citation_graph.values())
        }
        
        # THÊM PERMISSIONS VÀO RESPONSE
        response_data['permissions'] = permissions
        response_data['userRole'] = user_role
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in calculate_with_algorithm: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare-algorithms', methods=['POST'])
def compare_algorithms():
    try:
        data = request.get_json()
        algorithms = data.get('algorithms', ['pagerank', 'hits'])
        author_names = data.get('authors', [])
        paper_identifiers = data.get('papers', [])
        input_mode = data.get('input_mode', 'authors')
        input_type = data.get('input_type', 'doi')
        damping_factor = data.get('damping_factor', 0.85)
        max_iterations = data.get('max_iterations', 100)
        user_role = data.get('user_role', 'researcher')
        
        if not sch:
            return jsonify({'error': 'Semantic Scholar API not available'}), 500
        
        # Build network
        if input_mode == 'authors':
            if not author_names or len(author_names) == 0:
                return jsonify({'error': 'Please provide at least one author name'}), 400
            papers, citation_graph = build_citation_network(author_names)
        else:
            if not paper_identifiers or len(paper_identifiers) == 0:
                return jsonify({'error': 'Please provide at least one paper'}), 400
            papers, citation_graph = build_citation_network_from_papers(
                paper_identifiers,
                max_citations=MAX_CITATIONS_PER_PAPER,
                input_type=input_type
            )
        
        if len(papers) == 0:
            return jsonify({'error': 'No papers found'}), 404
        
        import time
        comparison_results = {}
        convergence_data = []

        # Run each algorithm
        for algo in algorithms:
            start_time = time.time()
            
            if algo == 'pagerank':
                results, residuals = calculate_pagerank(papers, citation_graph, damping_factor, max_iterations)
                comparison_results['pagerank'] = {
                    'results': results[:50],
                    'performance': {
                        'computation_time': round(time.time() - start_time, 3),
                        'iterations': len(residuals),
                        'papers_analyzed': len(papers)
                    }
                }
                convergence_data.append({
                    'algorithm': 'pagerank',
                    'residuals': residuals
                })
                
            elif algo == 'hits':
                authority_results, hub_results, iterations, residuals = calculate_hits(papers, citation_graph, max_iterations)
                comparison_results['hits'] = {
                    'authority_results': authority_results[:50],
                    'hub_results': hub_results[:50],
                    'performance': {
                        'computation_time': round(time.time() - start_time, 3),
                        'iterations': iterations,
                        'papers_analyzed': len(papers)
                    }
                }
                convergence_data.append({
                    'algorithm': 'hits',
                    'residuals': residuals
                })
                
            elif algo == 'weighted_pagerank':
                results, iterations, residuals = calculate_weighted_pagerank(papers, citation_graph, damping_factor, max_iterations)
                comparison_results['weighted_pagerank'] = {
                    'results': results[:50],
                    'performance': {
                        'computation_time': round(time.time() - start_time, 3),
                        'iterations': iterations,
                        'papers_analyzed': len(papers)
                    }
                }
                convergence_data.append({
                    'algorithm': 'weighted_pagerank',
                    'residuals': residuals
                })
        
        # Calculate correlations
        correlations = {}
        overlaps = {}
        
        if 'pagerank' in comparison_results and 'hits' in comparison_results:
            pr_results = comparison_results['pagerank']['results']
            hits_auth = comparison_results['hits']['authority_results']
            
            correlations['pagerank_vs_hits_authority'] = round(
                calculate_correlation(pr_results, hits_auth, 'pagerank', 'authority_score'), 3
            )
            overlaps['pagerank_vs_hits_top10'] = round(
                calculate_top_k_overlap(pr_results, hits_auth, 10), 3
            )
        
        if 'pagerank' in comparison_results and 'weighted_pagerank' in comparison_results:
            pr_results = comparison_results['pagerank']['results']
            wpr_results = comparison_results['weighted_pagerank']['results']
            
            correlations['pagerank_vs_weighted'] = round(
                calculate_correlation(pr_results, wpr_results, 'pagerank', 'weighted_pagerank'), 3
            )
            overlaps['pagerank_vs_weighted_top10'] = round(
                calculate_top_k_overlap(pr_results, wpr_results, 10), 3
            )
        
        # Prepare network data
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
        
        # LẤY PERMISSIONS THỰC TẾ
        permissions = USER_ROLES.get(user_role, USER_ROLES['researcher'])['permissions']

        response_data = {
            'algorithms': comparison_results,
            'correlations': correlations,
            'overlaps': overlaps,
            'convergence': convergence_data,
            'network': {
                'nodes': nodes,
                'edges': edges
            },
            'stats': {
                'totalPapers': len(papers),
                'totalCitations': sum(len(targets) for targets in citation_graph.values())
            },
            'permissions': permissions,
            'userRole': user_role
        }
        
        # CHỈ THÊM METRICS NẾU CÓ QUYỀN (cho comparison mode)
        if permissions.get('view_network_metrics', False):
            # Lấy results đầu tiên để tính metrics
            first_algo = algorithms[0]
            if first_algo in comparison_results:
                first_results = comparison_results[first_algo].get('results') or comparison_results[first_algo].get('authority_results')
                if first_results:
                    network_metrics = calculate_network_metrics_simple(papers, citation_graph, first_results)
                    response_data['networkMetrics'] = network_metrics
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in compare_algorithms: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/calculate-pagerank-stream', methods=['POST'])
def calculate_pagerank_stream():
    """
    SSE endpoint để tính PageRank với real-time progress updates
    Hỗ trợ cả input mode: authors và papers (DOI/title)
    """
    def generate():
        try:
            # Parse request data
            data = request.get_json()
            input_mode = data.get('input_mode', 'authors')
            author_names = data.get('authors', [])
            paper_identifiers = data.get('papers', [])
            input_type = data.get('input_type', 'doi')
            damping_factor = data.get('damping_factor', 0.85)
            max_iterations = data.get('max_iterations', 100)
            user_role = data.get('user_role', 'researcher')
            
            # Validation
            if input_mode == 'authors':
                if not author_names or len(author_names) == 0:
                    yield send_progress('error', 0, '❌ Please provide at least one author name')
                    return
            else:
                if not paper_identifiers or len(paper_identifiers) == 0:
                    yield send_progress('error', 0, f'❌ Please provide at least one paper {input_type}')
                    return
            
            if not sch:
                yield send_progress('error', 0, '❌ Semantic Scholar API not available')
                return
            
            # Send initial progress
            yield send_progress('starting', 0, '🚀 Initializing...')
            
            logger.info(f"Starting PageRank calculation (Mode: {input_mode}, Role: {user_role})")
            
            # Build citation network with progress
            all_papers = None
            citation_graph = None
            
            if input_mode == 'authors':
                for item in build_citation_network_with_progress(author_names):
                    if isinstance(item, tuple):
                        all_papers, citation_graph = item
                    else:
                        yield item
            else:
                for item in build_citation_network_from_papers_with_progress(
                    paper_identifiers,
                    max_citations=MAX_CITATIONS_PER_PAPER,
                    input_type=input_type
                ):
                    if isinstance(item, tuple):
                        all_papers, citation_graph = item
                    else:
                        yield item
            
            if not all_papers or len(all_papers) == 0:
                yield send_progress('error', 0, '❌ No papers found')
                return
            
            # Calculate PageRank
            yield send_progress('calculating', 85, '🧮 Calculating PageRank...')
            logger.info("Calculating PageRank...")
            
            results, residuals = calculate_pagerank(all_papers, citation_graph, damping_factor, max_iterations)
            
            yield send_progress('calculating_complete', 90, '✅ PageRank calculated!')
            
            # Prepare network data
            yield send_progress('preparing', 95, '📊 Preparing results...')
            logger.info("Preparing network data...")
            
            nodes = []
            edges = []
            
            for paper_id, paper in all_papers.items():
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
            
            # Get permissions
            permissions = USER_ROLES.get(user_role, USER_ROLES['researcher'])['permissions']
            
            # Build response
            response_data = {
                'status': 'complete',
                'progress': 100,
                'message': '✅ Complete!',
                'results': results[:50],
                'network': {
                    'nodes': nodes,
                    'edges': edges
                },
                'stats': {
                    'totalPapers': len(all_papers),
                    'totalCitations': sum(len(targets) for targets in citation_graph.values())
                },
                'userRole': user_role,
                'permissions': permissions
            }
            
            # Add network metrics if permitted
            if permissions.get('view_network_metrics', False):
                yield send_progress('calculating_metrics', 98, '📈 Calculating network metrics...')
                network_metrics = calculate_network_metrics_simple(all_papers, citation_graph, results)
                response_data['networkMetrics'] = network_metrics
            
            # Send final result
            yield f"data: {json.dumps(response_data)}\n\n"
            logger.info("✅ Stream complete!")
            
        except Exception as e:
            logger.error(f"Error in stream: {str(e)}")
            import traceback
            traceback.print_exc()
            yield send_progress('error', 0, f'❌ Error: {str(e)}')
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/calculate-citation-pagerank-stream', methods=['POST'])
def calculate_citation_pagerank_stream():
    """
    API endpoint với real-time progress updates sử dụng Server-Sent Events
    """
    def generate():
        try:
            # Parse request data
            data_json = request.get_json()
            author_names = data_json.get('authors', [])
            damping_factor = data_json.get('damping_factor', 0.85)
            max_iterations = data_json.get('max_iterations', 100)
            user_role = data_json.get('user_role', 'researcher')
            
            # Validation
            if not author_names or len(author_names) == 0:
                yield f"data: {json.dumps({'status': 'error', 'error': 'Please provide at least one author name'})}\n\n"
                return
            
            if not sch:
                yield f"data: {json.dumps({'status': 'error', 'error': 'Semantic Scholar API not available'})}\n\n"
                return
            
            # 🎯 BƯỚC 1: Khởi tạo (0-5%)
            yield f"data: {json.dumps({'status': 'starting', 'progress': 0, 'message': '🚀 Initializing...', 'stage': 'init'})}\n\n"
            
            all_papers = {}
            citation_graph = {}
            total_authors = len(author_names)
            
            # 🎯 BƯỚC 2: Fetch papers từ authors (5-35%)
            yield f"data: {json.dumps({'status': 'fetching_authors', 'progress': 5, 'message': f'👤 Fetching papers from {total_authors} authors...', 'stage': 'authors'})}\n\n"
            
            for idx, author_name in enumerate(author_names, 1):
                progress = 5 + int((idx / total_authors) * 30)  # 5% -> 35%
                
                yield f"data: {json.dumps({'status': 'fetching_author', 'progress': progress, 'message': f'[{idx}/{total_authors}] 🔍 Searching: {author_name}', 'current_author': author_name, 'author_index': idx, 'total_authors': total_authors})}\n\n"
                
                papers = search_papers_by_author(author_name, limit=10)
                
                logger.info(f"   ✅ Found {len(papers)} papers from {author_name}")
                
                for paper in papers:
                    paper_id = paper['paperId']
                    if paper_id not in all_papers:
                        all_papers[paper_id] = paper
                        citation_graph[paper_id] = []
                
                yield f"data: {json.dumps({'status': 'author_complete', 'progress': progress, 'message': f'✅ Found {len(papers)} papers from {author_name}', 'papers_found': len(papers)})}\n\n"
            
            # 🎯 BƯỚC 3: Fetch citations (35-85%)
            paper_ids_to_process = list(all_papers.keys())[:MAX_PAPERS_TO_PROCESS]
            total_papers = len(paper_ids_to_process)
            
            yield f"data: {json.dumps({'status': 'fetching_citations', 'progress': 35, 'message': f'🔗 Fetching citations for {total_papers} papers...', 'stage': 'citations', 'total_papers': total_papers})}\n\n"
            
            for idx, paper_id in enumerate(paper_ids_to_process, 1):
                progress = 35 + int((idx / total_papers) * 50)  # 35% -> 85%
                paper_title = all_papers[paper_id]['title'][:50]
                
                yield f"data: {json.dumps({'status': 'fetching_citation', 'progress': progress, 'message': f'[{idx}/{total_papers}] 📄 {paper_title}...', 'current_paper': idx, 'total_papers': total_papers})}\n\n"
                
                citations = get_paper_citations(paper_id, max_citations=MAX_CITATIONS_PER_PAPER)
                
                logger.info(f"   ✅ {len(citations)} citations")
                
                for cited_paper in citations:
                    cited_id = cited_paper['paperId']
                    if cited_id not in all_papers:
                        all_papers[cited_id] = cited_paper
                        citation_graph[cited_id] = []
                    
                    if paper_id not in citation_graph[cited_id]:
                        citation_graph[cited_id].append(paper_id)
            
            # 🎯 BƯỚC 4: Calculate PageRank (85-95%)
            yield f"data: {json.dumps({'status': 'calculating', 'progress': 85, 'message': '🧮 Calculating PageRank algorithm...', 'stage': 'calculation'})}\n\n"
            
            results, residuals = calculate_pagerank(all_papers, citation_graph, damping_factor, max_iterations)
            
            # 🎯 BƯỚC 5: Prepare results (95-100%)
            yield f"data: {json.dumps({'status': 'preparing', 'progress': 95, 'message': '📊 Preparing visualization data...', 'stage': 'finalize'})}\n\n"
            
            # Prepare network data
            nodes = []
            edges = []
            
            for paper_id, paper in all_papers.items():
                nodes.append({
                    'id': paper_id,
                    'label': paper['title'][:50] + '...' if len(paper['title']) > 50 else paper['title'],
                    'citationCount': paper['citationCount']
                })
            
            for source_id, targets in citation_graph.items():
                for target_id in targets:
                    edges.append({'from': source_id, 'to': target_id})
            
            # Get permissions
            permissions = USER_ROLES.get(user_role, USER_ROLES['researcher'])['permissions']
            
            # Build response
            response_data = {
                'status': 'complete',
                'progress': 100,
                'message': '✅ Complete! Analysis finished successfully.',
                'results': results[:50],
                'network': {'nodes': nodes, 'edges': edges},
                'stats': {
                    'totalPapers': len(all_papers),
                    'totalCitations': sum(len(targets) for targets in citation_graph.values())
                },
                'userRole': user_role,
                'permissions': permissions
            }
            
            # Add network metrics if permitted
            if permissions.get('view_network_metrics', False):
                network_metrics = calculate_network_metrics_simple(all_papers, citation_graph, results)
                response_data['networkMetrics'] = network_metrics
            
            # 🎯 BƯỚC 6: Send final result
            yield f"data: {json.dumps(response_data)}\n\n"
            
            logger.info(f"✅ Stream completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in stream: {str(e)}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'status': 'error', 'error': str(e), 'progress': 0})}\n\n"
    
    # Return SSE response
    return Response(
        stream_with_context(generate()), 
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)