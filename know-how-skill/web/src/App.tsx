import { Routes, Route, NavLink } from 'react-router-dom'
import WorkflowPage from './pages/WorkflowPage'
import BatchTestPage from './pages/BatchTestPage'
import VersionsPage from './pages/VersionsPage'

const navItems = [
  { to: '/', label: '增量修正' },
  { to: '/batch', label: '批量回测' },
  { to: '/versions', label: '版本管理' },
]

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-8">
        <h1 className="text-lg font-bold text-gray-800 whitespace-nowrap">
          Know-How 增量训练
        </h1>
        <nav className="flex gap-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-50 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>

      <main className="flex-1">
        <Routes>
          <Route path="/" element={<WorkflowPage />} />
          <Route path="/batch" element={<BatchTestPage />} />
          <Route path="/versions" element={<VersionsPage />} />
        </Routes>
      </main>
    </div>
  )
}
