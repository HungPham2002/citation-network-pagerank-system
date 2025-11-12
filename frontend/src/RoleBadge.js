import React from 'react';
import './RoleBadge.css';

function RoleBadge({ role, onChangeRole }) {
  const roleInfo = {
    researcher: {
      icon: '🔬',
      name: 'Researcher',
      color: '#0047AB'
    },
    data_scientist: {
      icon: '📊',
      name: 'Data Scientist',
      color: '#FF6B35'
    }
  };

  const info = roleInfo[role] || roleInfo.researcher;

  return (
    <div className="role-badge-container">
      <div className="role-badge" style={{ borderColor: info.color }}>
        <span className="role-icon">{info.icon}</span>
        <span className="role-name">{info.name}</span>
      </div>
      <button 
        className="btn-change-role"
        onClick={onChangeRole}
        title="Change your role"
      >
        Change Role
      </button>
    </div>
  );
}

export default RoleBadge;