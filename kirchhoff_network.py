"""
Sistema de tres nodos con resistencias - Leyes de Kirchhoff
Análisis de voltajes en una red resistiva usando eliminación de Gauss

Planteamiento del sistema Ax = b:
Las leyes de Kirchhoff (Kirchhoff's Voltage Law - KVL) para una red de tres nodos dan:

3V₁ - V₂ - V₃ = 5
-V₁ + 4V₂ - 2V₃ = 6
-V₁ - V₂ + 5V₃ = 7

Donde:
- V₁, V₂, V₃ son los voltajes en los tres nodos
- La matriz A contiene los coeficientes de la red
- El vector b contiene los voltajes o corrientes impuestos

Matriz A (coeficientes):
A = [  3  -1  -1 ]
    [ -1   4  -2 ]
    [ -1  -1   5 ]

Vector b (términos independientes):
b = [ 5 ]
    [ 6 ]
    [ 7 ]

Vector x (incógnitas - voltajes):
x = [ V₁ ]
    [ V₂ ]
    [ V₃ ]
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


class KirchhoffNetworkSolver:
    """Resuelve sistemas de ecuaciones de redes resistivas usando Leyes de Kirchhoff"""
    
    def __init__(self):
        """Inicializa el sistema de tres nodos"""
        # Matriz de coeficientes (Leyes de Kirchhoff)
        self.A = np.array([
            [3, -1, -1],
            [-1, 4, -2],
            [-1, -1, 5]
        ], dtype=float)
        
        # Vector de términos independientes (voltajes/corrientes impuestos)
        self.b = np.array([5, 6, 7], dtype=float)
        
        # Variables
        self.V = None  # Voltajes en los nodos
        
    def display_system(self):
        """Muestra el sistema Ax = b en forma clara"""
        print("\n" + "="*70)
        print("SISTEMA DE ECUACIONES - LEYES DE KIRCHHOFF")
        print("="*70)
        
        print("\nEcuaciones originales:")
        print("  3V₁ - V₂ - V₃ = 5")
        print("  -V₁ + 4V₂ - 2V₃ = 6")
        print("  -V₁ - V₂ + 5V₃ = 7")
        
        print("\n" + "-"*70)
        print("Forma matricial: Ax = b")
        print("-"*70)
        
        print("\nMatriz A (coeficientes):")
        print(self.A)
        
        print("\nVector b (términos independientes):")
        print(self.b)
        
        print("\nVector x (incógnitas - voltajes):")
        print("x = [V₁, V₂, V₃]ᵀ")
        
        print("\n" + "-"*70)
        print("Matriz aumentada [A|b]:")
        print("-"*70)
        
        augmented = np.column_stack((self.A, self.b))
        for i in range(len(self.A)):
            row_str = "["
            for j in range(len(self.A[i])):
                row_str += f"{augmented[i, j]:6.1f} "
            row_str += "| "
            row_str += f"{augmented[i, -1]:6.1f}]"
            print(row_str)
    
    def analyze_matrix(self):
        """Analiza las propiedades de la matriz A"""
        print("\n" + "="*70)
        print("ANÁLISIS DE LA MATRIZ A")
        print("="*70)
        
        # Determinante
        det_A = np.linalg.det(self.A)
        print(f"\nDeterminante de A: {det_A:.6f}")
        
        if np.abs(det_A) < 1e-10:
            print("⚠ Matriz singular (determinante ≈ 0)")
            print("  El sistema puede no tener solución única")
        else:
            print("✓ Matriz no singular")
            print("  El sistema tiene solución única")
        
        # Número de condición
        cond_A = np.linalg.cond(self.A)
        print(f"\nNúmero de condición: {cond_A:.6f}")
        
        if cond_A < 1e2:
            print("✓ Matriz bien condicionada (fácil de resolver)")
        elif cond_A < 1e4:
            print("⚠ Matriz moderadamente condicionada")
        else:
            print("⚠ Matriz mal condicionada (pueden haber errores de precisión)")
        
        # Autovalores
        eigenvalues = np.linalg.eigvals(self.A)
        print(f"\nAutovalores de A:")
        for i, eig in enumerate(eigenvalues):
            print(f"  λ{i+1} = {eig:.6f}")
        
        # Norma de la matriz
        norm_A = np.linalg.norm(self.A)
        print(f"\nNorma de Frobenius: {norm_A:.6f}")
        
        # Rango
        rank_A = np.linalg.matrix_rank(self.A)
        print(f"Rango de A: {rank_A}")
        print(f"Dimensión de A: {self.A.shape}")
        
        if rank_A == self.A.shape[0]:
            print("✓ Matriz de rango completo (todas las ecuaciones son independientes)")


class GaussianEliminationDetailed:
    """Implementa eliminación de Gauss con detalles paso a paso"""
    
    def __init__(self, A: np.ndarray, b: np.ndarray):
        """Inicializa con la matriz aumentada"""
        self.A = A.copy().astype(float)
        self.b = b.copy().astype(float)
        self.n = len(A)
        self.steps = []
    
    def forward_elimination(self):
        """Fase de eliminación hacia adelante"""
        print("\n" + "="*70)
        print("FASE 1: ELIMINACIÓN HACIA ADELANTE (Reducción a matriz triangular)")
        print("="*70)
        
        print("\nMatriz aumentada inicial:")
        self._print_augmented()
        
        for i in range(self.n):
            print(f"\n{'─'*70}")
            print(f"Paso {i+1}: Procesar columna {i+1}")
            print(f"{'─'*70}")
            
            # Pivoteo parcial
            pivot_row = i + np.argmax(np.abs(self.A[i:, i]))
            
            if pivot_row != i:
                print(f"\nPivoteo: Intercambiar fila {i+1} con fila {pivot_row+1}")
                print(f"(porque |{self.A[pivot_row, i]:.6f}| > |{self.A[i, i]:.6f}|)")
                
                self.A[[i, pivot_row]] = self.A[[pivot_row, i]]
                self.b[[i, pivot_row]] = self.b[[pivot_row, i]]
                
                self._print_augmented()
            
            # Verificar pivote
            if np.abs(self.A[i, i]) < 1e-10:
                print(f"⚠ Pivote cero en posición ({i}, {i})")
                continue
            
            # Eliminación
            print(f"\nEliminar elementos debajo del pivote (posición {i+1})")
            
            for j in range(i + 1, self.n):
                factor = self.A[j, i] / self.A[i, i]
                
                if np.abs(factor) > 1e-10:
                    print(f"\n  Fila {j+1} = Fila {j+1} - ({factor:.6f}) × Fila {i+1}")
                    
                    self.A[j, :] -= factor * self.A[i, :]
                    self.b[j] -= factor * self.b[i]
            
            self._print_augmented()
        
        return self.A, self.b
    
    def back_substitution(self) -> np.ndarray:
        """Fase de sustitución hacia atrás"""
        print("\n" + "="*70)
        print("FASE 2: SUSTITUCIÓN HACIA ATRÁS (Resolver el sistema triangular)")
        print("="*70)
        
        x = np.zeros(self.n)
        
        for i in range(self.n - 1, -1, -1):
            print(f"\n{'─'*70}")
            print(f"Ecuación {i+1}:")
            print(f"{'─'*70}")
            
            if np.abs(self.A[i, i]) < 1e-10:
                print(f"⚠ Coeficiente diagonal cero")
                x[i] = 0
                continue
            
            # Construir la ecuación
            suma = self.b[i]
            print(f"  {self.A[i, i]:.6f}×V{i+1} ", end="")
            
            for j in range(i + 1, self.n):
                if np.abs(self.A[i, j]) > 1e-10:
                    print(f"+ ({self.A[i, j]:.6f})×V{j+1} ", end="")
            
            print(f"= {self.b[i]:.6f}")
            
            # Restar términos conocidos
            print(f"\n  {self.A[i, i]:.6f}×V{i+1} = {self.b[i]:.6f}", end="")
            
            for j in range(i + 1, self.n):
                if np.abs(self.A[i, j]) > 1e-10:
                    print(f" - ({self.A[i, j]:.6f})×{x[j]:.6f}", end="")
                    suma -= self.A[i, j] * x[j]
            
            x[i] = suma / self.A[i, i]
            
            print(f"\n  {self.A[i, i]:.6f}×V{i+1} = {suma:.6f}")
            print(f"  V{i+1} = {x[i]:.10f}")
        
        return x
    
    def solve(self) -> np.ndarray:
        """Resuelve el sistema completo"""
        print("\n" + "#"*70)
        print("SOLUCIÓN POR ELIMINACIÓN DE GAUSS")
        print("#"*70)
        
        self.forward_elimination()
        x = self.back_substitution()
        
        return x
    
    def _print_augmented(self):
        """Imprime la matriz aumentada actual"""
        print("\nMatriz aumentada [A|b]:")
        for i in range(self.n):
            row_str = "[ "
            for j in range(len(self.A[i])):
                row_str += f"{self.A[i, j]:10.6f} "
            row_str += "| "
            row_str += f"{self.b[i]:10.6f} ]"
            print(row_str)


def solve_kirchhoff_gauss(solver: KirchhoffNetworkSolver) -> np.ndarray:
    """Resuelve el sistema de Kirchhoff usando eliminación de Gauss"""
    
    gauss_solver = GaussianEliminationDetailed(solver.A, solver.b)
    V = gauss_solver.solve()
    
    return V


def solve_kirchhoff_numpy(solver: KirchhoffNetworkSolver) -> np.ndarray:
    """Resuelve el sistema usando numpy.linalg.solve"""
    
    print("\n" + "="*70)
    print("SOLUCIÓN USANDO numpy.linalg.solve")
    print("="*70)
    
    try:
        V = np.linalg.solve(solver.A, solver.b)
        
        print(f"\n✓ Solución encontrada:")
        print(f"  V₁ = {V[0]:.10f} V")
        print(f"  V₂ = {V[1]:.10f} V")
        print(f"  V₃ = {V[2]:.10f} V")
        
        return V
    
    except np.linalg.LinAlgError as e:
        print(f"\n✗ Error: {e}")
        return None


def verify_solution(A: np.ndarray, b: np.ndarray, V: np.ndarray):
    """Verifica que la solución satisface el sistema"""
    
    print("\n" + "="*70)
    print("VERIFICACIÓN DE LA SOLUCIÓN")
    print("="*70)
    
    # Verificar Ax = b
    result = A @ V
    residual = result - b
    
    print(f"\nSolución: V = {V}")
    
    print(f"\nVerificando A×V = b:")
    print(f"  A×V = {result}")
    print(f"  b   = {b}")
    print(f"  Diferencia = {residual}")
    
    norm_residual = np.linalg.norm(residual)
    print(f"\nNorma del residual: {norm_residual:.2e}")
    
    if norm_residual < 1e-10:
        print("✓ ¡Solución verificada correctamente!")
        return True
    else:
        print("⚠ Residual significativo")
        return False


def print_equations_satisfied(A: np.ndarray, b: np.ndarray, V: np.ndarray):
    """Muestra cada ecuación verificada"""
    
    print("\n" + "="*70)
    print("ECUACIONES SATISFECHAS")
    print("="*70)
    
    variable_names = ['V₁', 'V₂', 'V₃']
    
    for i in range(len(A)):
        print(f"\nEcuación {i+1}:")
        
        # Construir la ecuación
        terms = []
        for j in range(len(A[i])):
            coeff = A[i, j]
            var = variable_names[j]
            if coeff >= 0 and len(terms) > 0:
                terms.append(f"+ {coeff:.1f}×{var}")
            else:
                terms.append(f"{coeff:.1f}×{var}")
        
        equation_str = " ".join(terms)
        result = np.dot(A[i], V)
        
        print(f"  {equation_str} = {result:.10f}")
        print(f"  Esperado: {b[i]:.10f}")
        print(f"  Error: {abs(result - b[i]):.2e}")
        
        if abs(result - b[i]) < 1e-10:
            print(f"  ✓ Verificado")
        else:
            print(f"  ⚠ Error detectado")


def plot_network_voltages(V: np.ndarray):
    """Grafica los voltajes en los nodos de la red"""
    
    print("\n" + "="*70)
    print("GENERANDO GRÁFICA DE VOLTAJES EN LOS NODOS")
    print("="*70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfica 1: Barras de voltaje
    nodes = ['Nodo 1\n(V₁)', 'Nodo 2\n(V₂)', 'Nodo 3\n(V₃)']
    colors = ['red' if v < 0 else 'blue' for v in V]
    
    bars = ax1.bar(nodes, V, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Voltaje (V)', fontsize=12, fontweight='bold')
    ax1.set_title('Voltajes en los Nodos', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for i, (bar, v) in enumerate(zip(bars, V)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{v:.4f} V',
                ha='center', va='bottom' if v > 0 else 'top',
                fontweight='bold', fontsize=11)
    
    # Gráfica 2: Red de nodos
    # Posiciones de los nodos en un triángulo
    node_positions = {
        0: (0, 1),      # V₁ arriba
        1: (-1, -0.5),  # V₂ abajo izquierda
        2: (1, -0.5)    # V₃ abajo derecha
    }
    
    for i, pos in node_positions.items():
        ax2.plot(pos[0], pos[1], 'o', markersize=30, color='lightblue', 
                edgecolor='black', linewidth=2, zorder=2)
        ax2.text(pos[0], pos[1], f'V{i+1}', ha='center', va='center',
                fontweight='bold', fontsize=12, zorder=3)
        ax2.text(pos[0], pos[1] - 0.3, f'{V[i]:.2f}V', ha='center', va='top',
                fontweight='bold', fontsize=10, color='darkblue', zorder=3)
    
    # Conexiones (resistencias)
    connections = [(0, 1), (0, 2), (1, 2)]
    for i, j in connections:
        pos_i = node_positions[i]
        pos_j = node_positions[j]
        ax2.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 'k-', linewidth=2, zorder=1)
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1, 1.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Topología de la Red', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('kirchhoff_network.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfica guardada como 'kirchhoff_network.png'")
    plt.show()


def main():
    """Función principal"""
    
    print("\n" + "#"*70)
    print("# SISTEMA DE TRES NODOS - LEYES DE KIRCHHOFF")
    print("#"*70)
    
    # Crear solver
    solver = KirchhoffNetworkSolver()
    
    # 1. Mostrar el sistema
    solver.display_system()
    
    # 2. Analizar la matriz
    solver.analyze_matrix()
    
    # 3. Resolver con Gauss
    V_gauss = solve_kirchhoff_gauss(solver)
    
    # 4. Resolver con numpy
    V_numpy = solve_kirchhoff_numpy(solver)
    
    # 5. Verificar solución
    verify_solution(solver.A, solver.b, V_numpy)
    
    # 6. Mostrar ecuaciones satisfechas
    print_equations_satisfied(solver.A, solver.b, V_numpy)
    
    # 7. Comparación
    print("\n" + "="*70)
    print("COMPARACIÓN DE RESULTADOS")
    print("="*70)
    print(f"\nElim. Gauss: V₁={V_gauss[0]:.10f}, V₂={V_gauss[1]:.10f}, V₃={V_gauss[2]:.10f}")
    print(f"numpy.solve: V₁={V_numpy[0]:.10f}, V₂={V_numpy[1]:.10f}, V₃={V_numpy[2]:.10f}")
    
    diff = np.linalg.norm(V_gauss - V_numpy)
    print(f"\nDiferencia (norma): {diff:.2e}")
    
    if diff < 1e-10:
        print("✓ Ambos métodos dan resultados idénticos")
    
    # 8. Graficar
    plot_network_voltages(V_numpy)
    
    # 9. Resumen
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"\n✓ Voltajes en los nodos:")
    print(f"  V₁ = {V_numpy[0]:.6f} V")
    print(f"  V₂ = {V_numpy[1]:.6f} V")
    print(f"  V₃ = {V_numpy[2]:.6f} V")
    print(f"\n✓ Sistema resuelto correctamente")


if __name__ == "__main__":
    main()
