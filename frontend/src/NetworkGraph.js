import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './NetworkGraph.css';

const NetworkGraph = ({ results, adjacencyMatrix }) => {
  const svgRef = useRef();
  const tooltipRef = useRef();

  useEffect(() => {
    if (!results || results.length === 0) return;

    // Clear previous graph
    d3.select(svgRef.current).selectAll('*').remove();

    // Setup dimensions
    const width = 900;
    const height = 600;

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height]);

    // Add zoom behavior
    const g = svg.append('g');
    
    const zoom = d3.zoom()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    svg.call(zoom);

    // Prepare data - UPDATED for citation network
    const nodes = results.map((result, i) => ({
      id: i,
      paperId: result.paperId,
      title: result.title,
      pagerank: result.pagerank,
      citationCount: result.citationCount,
      authors: result.authors,
      year: result.year,
      shortTitle: getShortTitle(result.title)
    }));

    // Calculate max/min pagerank for scaling
    const maxRank = Math.max(...results.map(r => r.pagerank));
    const minRank = Math.min(...results.map(r => r.pagerank));

    // Create links from edges (if provided)
    const links = [];
    if (adjacencyMatrix && adjacencyMatrix.length > 0) {
      // If adjacency matrix is provided as 2D array
      if (Array.isArray(adjacencyMatrix[0])) {
        for (let i = 0; i < adjacencyMatrix.length; i++) {
          for (let j = 0; j < adjacencyMatrix[i].length; j++) {
            if (adjacencyMatrix[i][j] > 0) {
              links.push({
                source: i,
                target: j,
                value: adjacencyMatrix[i][j]
              });
            }
          }
        }
      } else {
        // If edges provided as array of {from, to}
        adjacencyMatrix.forEach(edge => {
          const sourceIdx = nodes.findIndex(n => n.paperId === edge.from);
          const targetIdx = nodes.findIndex(n => n.paperId === edge.to);
          if (sourceIdx !== -1 && targetIdx !== -1) {
            links.push({
              source: sourceIdx,
              target: targetIdx,
              value: 1
            });
          }
        });
      }
    }

    // Calculate in-degree and out-degree
    const inDegree = new Array(nodes.length).fill(0);
    const outDegree = new Array(nodes.length).fill(0);
    links.forEach(link => {
      outDegree[typeof link.source === 'object' ? link.source.id : link.source]++;
      inDegree[typeof link.target === 'object' ? link.target.id : link.target]++;
    });

    // Color scale based on PageRank
    const colorScale = d3.scaleSequential()
      .domain([minRank, maxRank])
      .interpolator(d3.interpolateRgb('#ff4444', '#4CAF50'));

    // Node size scale
    const radiusScale = d3.scaleSqrt()
      .domain([minRank, maxRank])
      .range([15, 45]);

    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(150))
      .force('charge', d3.forceManyBody().strength(-800))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => radiusScale(d.pagerank) + 10));

    // Create arrow markers for directed edges
    svg.append('defs').selectAll('marker')
      .data(['arrow'])
      .join('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#999');

    // Draw links
    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', d => Math.sqrt(d.value) * 2)
      .attr('marker-end', 'url(#arrow)');

    // Draw nodes
    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .call(drag(simulation));

    // Add circles for nodes
    node.append('circle')
      .attr('r', d => radiusScale(d.pagerank))
      .attr('fill', d => colorScale(d.pagerank))
      .attr('stroke', '#fff')
      .attr('stroke-width', 3)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        // Highlight node
        d3.select(this)
          .attr('stroke', '#FFD700')
          .attr('stroke-width', 5);

        // Highlight connected links
        link
          .style('stroke-opacity', l => 
            (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.1
          )
          .style('stroke-width', l => 
            (l.source.id === d.id || l.target.id === d.id) ? 4 : 2
          );

        // Show tooltip
        const tooltip = d3.select(tooltipRef.current);
        tooltip.style('opacity', 1)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
          .html(`
            <div class="tooltip-content">
              <strong style="color: #0047AB; font-size: 1.1em;">📄 ${d.title.substring(0, 60)}${d.title.length > 60 ? '...' : ''}</strong>
              <hr style="margin: 8px 0; border-color: #e8f0fe;">
              <div style="display: grid; gap: 6px;">
                <div><strong>Authors:</strong> ${d.authors.slice(0, 3).join(', ')}${d.authors.length > 3 ? ' et al.' : ''}</div>
                <div><strong>Year:</strong> ${d.year || 'N/A'}</div>
                <div><strong>PageRank:</strong> ${d.pagerank.toFixed(6)}</div>
                <div><strong>Citations:</strong> ${d.citationCount || 0}</div>
                <div><strong>In-degree:</strong> ${inDegree[d.id]} citations</div>
                <div><strong>Out-degree:</strong> ${outDegree[d.id]} references</div>
                <div><strong>Rank:</strong> #${results.findIndex(r => r.paperId === d.paperId) + 1} of ${results.length}</div>
              </div>
            </div>
          `);
      })
      .on('mouseout', function(event, d) {
        // Reset node style
        d3.select(this)
          .attr('stroke', '#fff')
          .attr('stroke-width', 3);

        // Reset links
        link
          .style('stroke-opacity', 0.6)
          .style('stroke-width', l => Math.sqrt(l.value) * 2);

        // Hide tooltip
        d3.select(tooltipRef.current).style('opacity', 0);
      })
      .on('click', (event, d) => {
        window.open(`https://www.semanticscholar.org/paper/${d.paperId}`, '_blank');
      });

    // Add ranking badge
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', 5)
      .attr('font-size', d => Math.max(10, radiusScale(d.pagerank) * 0.4))
      .attr('font-weight', 'bold')
      .attr('fill', '#fff')
      .attr('pointer-events', 'none')
      .text((d, i) => `#${results.findIndex(r => r.paperId === d.paperId) + 1}`);

    // Add labels below nodes
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', d => radiusScale(d.pagerank) + 18)
      .attr('font-size', 11)
      .attr('font-weight', '600')
      .attr('fill', '#0047AB')
      .attr('pointer-events', 'none')
      .text(d => d.shortTitle);

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Drag functions
    function drag(simulation) {
      function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }

      function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
      }

      function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }

      return d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended);
    }

    // Helper function to shorten title
    function getShortTitle(title) {
      if (!title) return 'Unknown';
      if (title.length <= 30) return title;
      return title.substring(0, 27) + '...';
    }

  }, [results, adjacencyMatrix]);

  if (!results || results.length === 0) {
    return null;
  }

  return (
    <div className="network-graph-container">
      <div className="graph-header">
        <h3>Citation Network Graph Visualization</h3>
        <div className="graph-controls">
          <div className="legend">
            <div className="legend-item">
              <div className="legend-color" style={{ background: 'linear-gradient(90deg, #ff4444, #4CAF50)' }}></div>
              <span>PageRank Score (Low → High)</span>
            </div>
            <div className="legend-item">
              <div className="legend-icon">⭕</div>
              <span>Node size = Paper importance</span>
            </div>
            <div className="legend-item">
              <div className="legend-icon">➡️</div>
              <span>Arrow = Citation direction</span>
            </div>
          </div>
        </div>
        <p className="graph-instructions">
          💡 <strong>Tip:</strong> Hover over nodes to see paper details | Click to view on Semantic Scholar | Drag nodes to rearrange | Scroll to zoom
        </p>
      </div>
      <svg ref={svgRef}></svg>
      <div ref={tooltipRef} className="graph-tooltip"></div>
    </div>
  );
};

export default NetworkGraph;