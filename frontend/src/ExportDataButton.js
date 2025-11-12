import React, { useState } from 'react';
import './ExportDataButton.css';

function ExportDataButton({ results, network, stats, networkMetrics, inputMode, parameters, userRole }) {
  const [exporting, setExporting] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);

  const handleExport = async (format) => {
    setExporting(true);
    setShowDropdown(false);

    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      
      const exportData = {
        user_role: userRole,
        format: format,
        results: results,
        network: network,
        stats: stats,
        networkMetrics: networkMetrics,
        input_mode: inputMode,
        parameters: parameters
      };

      const response = await fetch(`${apiUrl}/api/export-data`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(exportData)
      });

      const data = await response.json();

      if (response.ok && data.success) {
        // Download file
        let blob;
        if (format === 'json') {
          blob = new Blob([JSON.stringify(data.data, null, 2)], { type: 'application/json' });
        } else {
          blob = new Blob([data.data], { type: 'text/csv' });
        }

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = data.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        alert(`✅ Data exported successfully as ${format.toUpperCase()}!`);
      } else {
        alert(`❌ Export failed: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      alert(`❌ Export error: ${error.message}`);
      console.error('Export error:', error);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="export-data-container">
      <button 
        className="btn-export-main"
        onClick={() => setShowDropdown(!showDropdown)}
        disabled={exporting}
      >
        {exporting ? (
          <>
            <span className="loading-spinner"></span>
            Exporting...
          </>
        ) : (
          <>
            Export Data
            <span className="dropdown-arrow">{showDropdown ? '▲' : '▼'}</span>
          </>
        )}
      </button>

      {showDropdown && !exporting && (
        <div className="export-dropdown">
          <div className="export-dropdown-header">
            Select Export Format
          </div>
          <button 
            className="export-option"
            onClick={() => handleExport('json')}
          >
            <span className="format-icon">{ }</span>
            <div className="format-info">
              <strong>JSON Format</strong>
              <small>Complete data structure for programming (Python, R, JavaScript)</small>
            </div>
          </button>
          <button 
            className="export-option"
            onClick={() => handleExport('csv')}
          >
            <span className="format-icon"></span>
            <div className="format-info">
              <strong>CSV Format</strong>
              <small>Table format for Excel, Google Sheets, Tableau</small>
            </div>
          </button>
        </div>
      )}
    </div>
  );
}

export default ExportDataButton;