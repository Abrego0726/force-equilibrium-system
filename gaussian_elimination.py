"""
Sistema de Equilibrio de Fuerzas en un Nodo Plano
Solución por Eliminación de Gauss (Gaussian Elimination)

Un nodo está sometido a tres fuerzas F1, F2, F3 con ángulos de 0°, 45° y 120°.
Se necesita encontrar las componentes que satisfacen el equilibrio:
    Fx1 + Fx2 + Fx3 = 0
    Fy1 + Fy2 + Fy3 = 0
"""

import numpy as np
from typing import Tuple, List


class GaussianElimination:
    """Clase para resolver sistemas de ecuaciones lineales usando eliminación de Gauss"""
    
    def __init__(self, A: np.ndarray, b: np.ndarray):
        """
        Inicializa el sistema Ax = b
        
        Args:
            A: Matriz de coeficientes (n x m)
            b: Vector de términos independientes (n x 1)
        """
        self.A_original = A.copy().astype(float)
        self.b_original = b.copy().astype(float)
        self.A = A.copy().astype(float)
        self.b = b.copy().astype(float)
        self.n = A.shape[0]  # número de ecuaciones
        self.m = A.shape[1]  # número de incógnitas
        self.steps = []  # Para registrar los pasos
        
    def print_system(self, title: str = "Sistema"):
        """Imprime el sistema actual en forma de matriz aumentada"""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        print(f"Matriz aumentada [A|b]:")
        
        # Crear matriz aumentada
        augmented = np.column_stack((self.A, self.b))
        
        for i in range(self.n):
            row_str = "| "
            for j in range(self.m):
                row_str += f"{augmented[i, j]:10.6f} "
            row_str += "| "
            row_str += f"{augmented[i, self.m]:10.6f} "
            print(row_str + "|")
        print()
    
    def forward_elimination(self):
        """
        Realiza la fase de eliminación hacia adelante (Forward Elimination)
        Convierte la matriz a forma triangular superior
        """
        print("\n" + "="*60)
        print("FASE 1: ELIMINACIÓN HACIA ADELANTE")
        print("="*60)
        
        self.print_system("Sistema inicial")
        
        for col in range(min(self.n, self.m)):
            # Paso 1: Encontrar el pivote (elemento más grande en valor absoluto)
            pivot_row = col + np.argmax(np.abs(self.A[col:, col]))
            
            # Paso 2: Intercambiar filas si es necesario (pivoteo parcial)
            if pivot_row != col:
                print(f"\nPaso {col+1}: Intercambiar fila {col+1} con fila {pivot_row+1}")
                self.A[[col, pivot_row]] = self.A[[pivot_row, col]]
                self.b[[col, pivot_row]] = self.b[[pivot_row, col]]
                self.print_system(f"Después del intercambio")
            
            # Verificar si el pivote es cero
            if np.abs(self.A[col, col]) < 1e-10:
                print(f"¡Pivote cero en posición ({col}, {col})!")
                continue
            
            # Paso 3: Eliminar los elementos debajo del pivote
            print(f"\nPaso {col+1}: Eliminar elementos en columna {col+1} debajo del pivote")
            
            for row in range(col + 1, self.n):
                if np.abs(self.A[col, col]) > 1e-10:
                    factor = self.A[row, col] / self.A[col, col]
                    
                    if np.abs(factor) > 1e-10:
                        print(f"  Fila {row+1} = Fila {row+1} - ({factor:.6f}) * Fila {col+1}")
                        self.A[row, :] -= factor * self.A[col, :]
                        self.b[row] -= factor * self.b[col]
            
            self.print_system(f"Después de eliminar columna {col+1}")
    
    def back_substitution(self) -> np.ndarray:
        """
        Realiza la fase de sustitución hacia atrás (Back Substitution)
        Resuelve el sistema desde la forma triangular superior
        
        Returns:
            Vector solución x
        """
        print("\n" + "="*60)
        print("FASE 2: SUSTITUCIÓN HACIA ATRÁS")
        print("="*60)
        
        x = np.zeros(self.m)
        
        # Resolver desde la última ecuación hasta la primera
        for i in range(min(self.n, self.m) - 1, -1, -1):
            if np.abs(self.A[i, i]) < 1e-10:
                print(f"\nEcuación {i+1}: Coeficiente diagonal cero. Sistema singular o indeterminado.")
                x[i] = 0  # Variable libre
                continue
            
            # Calcular x[i]
            suma = self.b[i]
            for j in range(i + 1, self.m):
                suma -= self.A[i, j] * x[j]
            
            x[i] = suma / self.A[i, i]
            
            print(f"\nEcuación {i+1}:")
            print(f"  {self.A[i, i]:.6f} * x[{i}] = {self.b[i]:.6f}", end="")
            
            if i < self.m - 1:
                terms = " - ".join([f"({self.A[i, j]:.6f} * {x[j]:.6f})" 
                                   for j in range(i + 1, self.m)])
                print(f" - ({terms})", end="")
            
            print(f"\n  x[{i}] = {x[i]:.6f}")
        
        return x
    
    def solve(self) -> np.ndarray:
        """
        Resuelve el sistema completo usando eliminación de Gauss
        
        Returns:
            Vector solución x
        """
        print("\n" + "#"*60)
        print("SOLUCIÓN DEL SISTEMA POR ELIMINACIÓN DE GAUSS")
        print("#"*60)
        
        self.forward_elimination()
        x = self.back_substitution()
        
        return x
    
    def verify_solution(self, x: np.ndarray):
        """
        Verifica que la solución es correcta
        
        Args:
            x: Vector solución
        """
        print("\n" + "="*60)
        print("VERIFICACIÓN DE LA SOLUCIÓN")
        print("="*60)
        
        residual = self.A_original @ x - self.b_original
        
        print(f"\nSolución x = {x}")
        print(f"\nA * x = {self.A_original @ x}")
        print(f"b     = {self.b_original}")
        print(f"\nResiduo (A*x - b) = {residual}")
        print(f"Norma del residuo = {np.linalg.norm(residual):.2e}")
        
        if np.linalg.norm(residual) < 1e-10:
            print("✓ ¡Solución verificada correctamente!")
        else:
            print("⚠ La solución tiene residuo significativo")


def solve_force_equilibrium(angles_deg: List[float], 
                            magnitudes: List[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resuelve el sistema de equilibrio de fuerzas en un nodo plano
    
    Args:
        angles_deg: Lista de ángulos en grados [θ1, θ2, θ3]
        magnitudes: Lista de magnitudes de fuerzas (opcional)
    
    Returns:
        Tupla (componentes_x, componentes_y) de las fuerzas
    """
    # Convertir ángulos a radianes
    angles_rad = np.radians(angles_deg)
    
    # Construir la matriz A con los cosenos y senos
    # Fila 1: componentes en x (cosenos)
    # Fila 2: componentes en y (senos)
    A = np.array([
        [np.cos(angles_rad[0]), np.cos(angles_rad[1]), np.cos(angles_rad[2])],
        [np.sin(angles_rad[0]), np.sin(angles_rad[1]), np.sin(angles_rad[2])]
    ])
    
    # Vector de términos independientes (equilibrio = 0)
    b = np.array([0.0, 0.0])
    
    print("\n" + "#"*60)
    print("SISTEMA DE EQUILIBRIO DE FUERZAS EN UN NODO PLANO")
    print("#"*60)
    print(f"\nÁngulos de las fuerzas: {angles_deg}°")
    print(f"Matriz de direcciones:\n{A}")
    print(f"Vector de equilibrio: {b}")
    
    # Resolver usando eliminación de Gauss
    solver = GaussianElimination(A, b)
    
    # Para un sistema homogéneo (2x3), la solución general es una familia de soluciones
    # Usaremos el método de Gauss para encontrar una base del espacio nulo
    
    # Si se proporcionan magnitudes específicas, resolver directamente
    if magnitudes is not None:
        # Usar la primera magnitud como restricción adicional
        # Resolver para F2 y F3 en términos de F1
        A_constrained = np.array([
            [np.cos(angles_rad[1]), np.cos(angles_rad[2])],
            [np.sin(angles_rad[1]), np.sin(angles_rad[2])]
        ])
        
        b_constrained = np.array([
            -magnitudes[0] * np.cos(angles_rad[0]),
            -magnitudes[0] * np.sin(angles_rad[0])
        ])
        
        solver_constrained = GaussianElimination(A_constrained, b_constrained)
        solution = solver_constrained.solve()
        
        # Construir la solución completa
        F = np.array([magnitudes[0], solution[0], solution[1]])
        
        print(f"\n✓ Magnitudes de fuerzas: {F}")
        
        # Calcular componentes
        Fx = np.array([F[i] * np.cos(angles_rad[i]) for i in range(3)])
        Fy = np.array([F[i] * np.sin(angles_rad[i]) for i in range(3)])
        
        print(f"Componentes en X: {Fx}")
        print(f"Componentes en Y: {Fy}")
        print(f"\nVerificación de equilibrio:")
        print(f"  ΣFx = {np.sum(Fx):.2e}")
        print(f"  ΣFy = {np.sum(Fy):.2e}")
        
        return Fx, Fy
    
    else:
        # Para sistema homogéneo, mostrar el espacio nulo
        print("\n⚠ Sistema homogéneo (2 ecuaciones, 3 incógnitas)")
        print("La solución general forma una recta en el espacio 3D")
        
        # Calcular el espacio nulo usando SVD
        U, s, Vt = np.linalg.svd(A)
        null_space = Vt[-1:]  # Última fila de Vt
        
        print(f"\nBase del espacio nulo: {null_space[0]}")
        print(f"Solución general: F = t * {null_space[0]}, donde t es cualquier escalar")
        
        return None, None


def example_with_specific_magnitudes():
    """Ejemplo con magnitudes específicas de fuerzas"""
    print("\n\n" + "="*60)
    print("EJEMPLO: Equilibrio con magnitudes específicas")
    print("="*60)
    
    angles = [0, 45, 120]  # Ángulos en grados
    F1 = 100  # Magnitud de F1 en Newtons
    
    print(f"\nCondiciones:")
    print(f"  - Ángulos: {angles}°")
    print(f"  - Magnitud de F1: {F1} N")
    print(f"  - Se buscan F2 y F3 para equilibrio")
    
    Fx, Fy = solve_force_equilibrium(angles, [F1, None, None])
    
    return Fx, Fy


if __name__ == "__main__":
    # Ejemplo 1: Sistema homogéneo
    print("\n" + "*"*60)
    print("EJEMPLO 1: Sistema homogéneo (sin magnitudes específicas)")
    print("*"*60)
    
    angles = [0, 45, 120]
    solve_force_equilibrium(angles)
    
    # Ejemplo 2: Con magnitudes específicas
    example_with_specific_magnitudes()
